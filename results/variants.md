# PPO-attention training variants

Validation: mean task return of each seed's `best.pt` on the 100 validation seeds (nominal), used to pick the default. Scenario tables: the 100 held-out seeds. ± is a 95% CI over training seeds.

- `ppo_attn_smdp`: original agent (SMDP γ^τ, lr 3e-4, best of 20-episode validation every 25 updates)
- `abl_no_smdp`: per-decision γ, otherwise as `ppo_attn_smdp`
- `ppo_attn`: per-decision γ, lr 1.5e-4 (annealed), 50-episode validation every 10 updates — **current default**
- `ppo_attn_big`: `ppo_attn` with d_model 128 and 3 layers (518k parameters)

| Variant | seeds | Validation task return |
|---|---|---|
| ppo_attn_smdp | 5 | 104.7 ± 9.5 |
| abl_no_smdp | 5 | 135.8 ± 2.0 |
| ppo_attn | 5 | 140.9 ± 2.2 |
| ppo_attn_big | 3 | 139.6 ± 4.8 |

## nominal

| Variant | Task return | Revenue % | On-time % | Express on-time % | Travel / order |
|---|---|---|---|---|---|
| ppo_attn_smdp | 132.0 ± 9.5 | 60.2 ± 1.8 | 45.2 ± 2.4 | 73.5 ± 1.9 | 40.1 ± 1.7 |
| abl_no_smdp | 158.1 ± 5.7 | 67.5 ± 0.7 | 52.2 ± 0.5 | 78.1 ± 2.3 | 38.5 ± 0.5 |
| ppo_attn | 153.0 ± 3.9 | 67.3 ± 0.8 | 52.3 ± 1.2 | 78.3 ± 1.5 | 38.6 ± 1.1 |
| ppo_attn_big | 151.4 ± 1.2 | 67.2 ± 0.9 | 52.8 ± 2.2 | 75.8 ± 2.4 | 38.8 ± 0.7 |

## ood_a

| Variant | Task return | Revenue % | On-time % | Express on-time % | Travel / order |
|---|---|---|---|---|---|
| ppo_attn_smdp | 135.0 ± 3.1 | 60.5 ± 0.9 | 47.2 ± 2.1 | 76.8 ± 0.7 | 39.1 ± 0.8 |
| abl_no_smdp | 159.7 ± 3.8 | 67.9 ± 0.9 | 54.7 ± 0.8 | 80.8 ± 0.6 | 36.7 ± 0.7 |
| ppo_attn | 164.2 ± 3.9 | 68.0 ± 0.7 | 54.4 ± 0.8 | 81.8 ± 1.7 | 36.5 ± 0.9 |
| ppo_attn_big | 167.5 ± 6.6 | 69.2 ± 0.7 | 55.5 ± 1.4 | 80.9 ± 4.9 | 36.2 ± 0.5 |

## ood_b

| Variant | Task return | Revenue % | On-time % | Express on-time % | Travel / order |
|---|---|---|---|---|---|
| ppo_attn_smdp | 98.3 ± 6.3 | 54.2 ± 1.5 | 42.1 ± 2.5 | 68.3 ± 1.9 | 43.6 ± 2.0 |
| abl_no_smdp | 120.7 ± 4.5 | 61.9 ± 0.9 | 49.5 ± 0.3 | 72.0 ± 1.6 | 43.1 ± 0.5 |
| ppo_attn | 126.9 ± 4.7 | 62.6 ± 1.2 | 49.2 ± 1.3 | 73.2 ± 1.4 | 42.6 ± 1.0 |
| ppo_attn_big | 130.8 ± 2.4 | 63.8 ± 0.7 | 50.8 ± 1.6 | 74.0 ± 4.2 | 42.0 ± 0.2 |

## rush_x1.5

| Variant | Task return | Revenue % | On-time % | Express on-time % | Travel / order |
|---|---|---|---|---|---|
| ppo_attn_smdp | 114.0 ± 15.8 | 46.5 ± 1.3 | 33.6 ± 0.7 | 63.3 ± 1.3 | 37.5 ± 1.4 |
| abl_no_smdp | 143.5 ± 9.1 | 51.6 ± 0.9 | 38.5 ± 0.7 | 68.4 ± 1.4 | 35.6 ± 0.4 |
| ppo_attn | 146.7 ± 5.3 | 51.9 ± 0.9 | 38.4 ± 1.1 | 68.2 ± 1.3 | 35.4 ± 1.0 |
| ppo_attn_big | 150.6 ± 8.4 | 52.9 ± 1.7 | 39.5 ± 1.6 | 67.8 ± 3.4 | 35.4 ± 0.9 |

## capacity_4

| Variant | Task return | Revenue % | On-time % | Express on-time % | Travel / order |
|---|---|---|---|---|---|
| ppo_attn_smdp | 104.2 ± 6.5 | 54.9 ± 0.9 | 40.8 ± 1.3 | 66.1 ± 0.9 | 45.3 ± 1.9 |
| abl_no_smdp | 127.6 ± 7.4 | 61.0 ± 0.8 | 46.4 ± 0.7 | 70.2 ± 2.0 | 44.4 ± 0.4 |
| ppo_attn | 127.5 ± 6.4 | 61.1 ± 0.9 | 46.3 ± 0.7 | 70.9 ± 1.8 | 44.2 ± 0.7 |
| ppo_attn_big | 130.8 ± 6.3 | 62.0 ± 1.5 | 47.1 ± 1.2 | 69.7 ± 1.5 | 44.1 ± 0.1 |
