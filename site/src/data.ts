// Types and loaders for the JSON written by experiments/export_site_data.py.

export type Cell = [number, number];

export interface Layout {
  name: string;
  grid: string[];
  docks: Cell[];
  charger: Cell | null;
  packing: Cell;
}

export interface Truck {
  id: number;
  dock: number;
  scheduled: number;
  actual: number;
  final: boolean;
}

export interface Item {
  cell: Cell;
  w: number;
  pick: number | null;
  load: number | null;
}

export interface Order {
  id: number;
  arrival: number;
  tier: 0 | 1 | 2;
  vip: boolean;
  value: number;
  captured: number;
  dock: number;
  items: Item[];
  truck0: number;
  truck: number;
  deferrals: number;
  status: "fulfilled" | "lost" | "closed";
  resolved: number;
}

export interface Option {
  r: number;
  k: "pick" | "deliver" | "charge" | "wait";
  t0: number;
  t1: number;
  travel: number;
  path: Cell[];
  bat: number;
  order?: number;
  item?: number;
  dock?: number;
  items?: [number, number][];
}

export interface Candidate {
  a: string;
  v: number;
}

export interface LookaheadDecision {
  t: number;
  cands: Candidate[];
  chosen: string;
  base: string;
}

export interface Episode {
  seed: number;
  horizon: number;
  robots: number;
  capacity: number;
  battery: number;
  layout: Layout;
  trucks: Truck[];
  orders: Order[];
  options: Option[];
  revenue: [number, number][];
  revenueTotal: number;
  rate: [number, number][];
  stats: Record<string, number>;
  attention?: { t: number; w: Record<string, number> }[];
  lookahead?: LookaheadDecision[];
}

export type Metrics = {
  revenue_captured_pct: number;
  task_return: number;
  on_time_pct: number;
  express_on_time_pct: number;
  travel_ticks_per_order: number;
};

export interface Summary {
  agents: Record<string, Record<string, Metrics>>;
  lookahead: { with: Metrics; without: Metrics };
  fleet: Record<string, Record<string, number>>;
  variants: Record<string, { validation: number; revenue: number; task_return: number }>;
  curves: Record<string, [number, number][][]>;
}

const base = import.meta.env.BASE_URL;

async function load<T>(name: string): Promise<T> {
  const res = await fetch(`${base}data/${name}.json`);
  if (!res.ok) throw new Error(`failed to load ${name}`);
  return res.json() as Promise<T>;
}

export const loadEpisode = (name: string) => load<Episode>(`episode_${name}`);
export const loadSummary = () => load<Summary>("summary");

export const TIER_NAMES = ["Express", "Standard", "Bulk"] as const;
