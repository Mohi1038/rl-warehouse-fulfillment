import numpy as np
import pytest
import torch

from agents.baselines import BASELINES, make_baseline
from agents.common import RunningMeanStd, baseline_act_batch, evaluate_policy, to_torch
from agents.dqn import NStepBuffer, PrioritizedReplay, SumTree
from agents.networks import AttentionActorCritic, AttentionQNetwork, MLPActorCritic
from agents.ppo import smdp_gae
from curriculum import Curriculum, CurriculumConfig
from warehouse_env import EnvConfig, MultiRobotWarehouseEnv, WarehouseEnv


def _batch(env, n=4, seed=0):
    rng = np.random.default_rng(seed)
    observations = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n * 6):
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        observations.append(obs)
    picked = observations[-n:]
    return to_torch({k: np.stack([o[k] for o in picked]) for k in picked[0]})


def test_attention_policy_is_permutation_equivariant():
    torch.manual_seed(0)
    env = WarehouseEnv(base_rate=0.08)
    net = AttentionActorCritic(n_docks=3).eval()
    obs = _batch(env)
    perm = torch.randperm(16)
    permuted = dict(obs, orders=obs["orders"][:, perm])
    logits, value, _ = net(obs)
    logits_p, value_p, _ = net(permuted)
    assert torch.allclose(logits[:, :16][:, perm], logits_p[:, :16], atol=1e-5)
    assert torch.allclose(logits[:, 16:], logits_p[:, 16:], atol=1e-5)
    assert torch.allclose(value, value_p, atol=1e-5)


def test_attention_policy_scales_to_more_slots():
    net = AttentionActorCritic(n_docks=3).eval()
    for slots in (16, 24, 32):
        obs = _batch(WarehouseEnv(max_orders=slots, base_rate=0.1))
        logits, value, attention = net(obs, need_weights=True)
        assert logits.shape == (4, slots + 5) and value.shape == (4,)
        assert attention.shape == (4, slots)
        valid = obs["orders"][..., 0] > 0.5
        assert torch.all(attention[~valid] < 1e-6)


def test_mlp_sorting_maps_logits_back_to_slots():
    torch.manual_seed(0)
    obs = _batch(WarehouseEnv(base_rate=0.08))
    sorted_net = MLPActorCritic(n_docks=3, n_orders=16).eval()
    perm = torch.randperm(16)
    permuted = dict(obs, orders=obs["orders"][:, perm])
    a, _, _ = sorted_net(obs)
    b, _, _ = sorted_net(permuted)
    valid = permuted["orders"][..., 0] > 0.5  # empty slots tie on the sort key; they are masked anyway
    assert valid.any()
    assert torch.allclose(a[:, :16][:, perm][valid], b[:, :16][valid], atol=1e-5)
    with pytest.raises(ValueError):
        sorted_net(_batch(WarehouseEnv(max_orders=24)))


def test_networks_with_robot_tokens():
    obs = _batch(MultiRobotWarehouseEnv(n_robots=2))
    for net in (AttentionActorCritic(3, use_robots=True), MLPActorCritic(3, 16, use_robots=True)):
        logits, value, _ = net(obs)
        assert logits.shape == (4, 21)
    q, _ = AttentionQNetwork(3, use_robots=True)(obs, mask=obs["action_mask"])
    assert q.shape == (4, 21)


def test_dueling_q_centres_advantages_over_valid_actions():
    obs = _batch(WarehouseEnv(base_rate=0.08))
    net = AttentionQNetwork(3).eval()
    mask = obs["action_mask"]
    q, _ = net(obs, mask=mask)
    h_cls, h_orders, valid, _ = net.encoder(obs)
    pooled = (h_orders * valid.float().unsqueeze(-1)).sum(1) / valid.float().sum(1, keepdim=True).clamp(min=1)
    value = net.value_head(torch.cat([h_cls, pooled], -1)).squeeze(-1)
    mean_valid_q = (q * mask).sum(-1) / mask.sum(-1)
    assert torch.allclose(mean_valid_q, value, atol=1e-5)


def test_smdp_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [0.4], [0.3]])
    dones = np.array([[0.0], [1.0], [0.0]])
    taus = np.array([[2.0], [1.0], [3.0]])
    last = np.array([0.7])
    adv, ret = smdp_gae(rewards, values, dones, taus, last, gamma, lam)
    d2 = 3.0 + gamma**3 * 0.7 - 0.3
    d1 = 2.0 - 0.4  # terminal: no bootstrap, no trace
    d0 = 1.0 + gamma**2 * 0.4 - 0.5
    assert adv[2, 0] == pytest.approx(d2)
    assert adv[1, 0] == pytest.approx(d1)
    assert adv[0, 0] == pytest.approx(d0 + (gamma * lam) ** 2 * d1)
    assert np.allclose(ret, adv + values)
    flat, _ = smdp_gae(rewards, values, dones, taus, last, gamma, lam, smdp=False)
    assert flat[2, 0] == pytest.approx(3.0 + gamma * 0.7 - 0.3)


def test_sum_tree_sampling_is_proportional():
    tree = SumTree(5)
    tree.update(np.arange(5), np.array([1.0, 0.0, 3.0, 0.0, 6.0]))
    assert tree.total == pytest.approx(10.0)
    rng = np.random.default_rng(0)
    leaves = tree.find(rng.random(20_000) * tree.total)
    counts = np.bincount(leaves, minlength=5) / 20_000
    assert np.allclose(counts, [0.1, 0, 0.3, 0, 0.6], atol=0.02)


def test_prioritized_replay_roundtrip():
    example = {"x": np.zeros(3, dtype=np.float32)}
    replay = PrioritizedReplay(8, example, alpha=0.6, eps=1e-3)
    for i in range(10):
        replay.add({"x": np.full(3, i, np.float32)}, i, float(i), 0.9, {"x": np.zeros(3, np.float32)})
    assert len(replay) == 8
    indices, obs, actions, *_, weights = replay.sample(4, beta=0.4, rng=np.random.default_rng(0))
    assert np.all(obs["x"][:, 0] == actions)
    assert weights.max() == pytest.approx(1.0)
    replay.update_priorities(indices, np.array([5.0, 0.0, 0.0, 0.0]))
    assert replay.max_priority >= 5.0


def test_n_step_buffer_uses_smdp_discounting():
    buffer = NStepBuffer(n=2, gamma=0.5)
    assert buffer.push("s0", 0, 1.0, 2.0, "s1", False) == []
    out = buffer.push("s1", 1, 4.0, 1.0, "s2", False)
    assert out == [("s0", 0, 1.0 + 0.5**2 * 4.0, 0.5**3, "s2")]
    out = buffer.push("s2", 2, 8.0, 1.0, "s3", True)
    assert [item[0] for item in out] == ["s1", "s2"] and all(item[3] == 0.0 for item in out)
    assert out[0][2] == pytest.approx(4.0 + 0.5 * 8.0)


def test_running_mean_std():
    rms = RunningMeanStd()
    data = np.random.default_rng(0).normal(3.0, 2.0, size=10_000)
    for chunk in np.split(data, 100):
        rms.update(chunk)
    assert rms.mean == pytest.approx(3.0, abs=0.1) and np.sqrt(rms.var) == pytest.approx(2.0, abs=0.1)


def test_curriculum_promotion_and_fallback():
    curriculum = Curriculum(CurriculumConfig(window=3, threshold=80, max_updates_per_level=5, max_level=2))
    for value in (90, 85, 95):
        curriculum.record(value)
    assert curriculum.step() and curriculum.level == 1
    for _ in range(4):
        assert not curriculum.step()
    assert curriculum.step() and curriculum.level == 2
    assert not curriculum.step()
    assert Curriculum(CurriculumConfig(enabled=False)).level is None


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_only_choose_valid_actions(name):
    config = EnvConfig()
    env = WarehouseEnv(config)
    results = evaluate_policy(baseline_act_batch(make_baseline(name, env)), config, seeds=range(3))
    assert all(r["invalid_actions"] == 0 for r in results)


def test_baselines_beat_random():
    config = EnvConfig()
    env = WarehouseEnv(config)
    score = {
        name: np.mean([r["return"] for r in evaluate_policy(baseline_act_batch(make_baseline(name, env)), config, range(10))])
        for name in ("random", "greedy")
    }
    assert score["greedy"] > score["random"]
