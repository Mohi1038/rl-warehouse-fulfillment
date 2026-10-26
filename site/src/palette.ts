// Colour tokens. Tier and robot sets are validated (dataviz validate_palette.js, dark mode,
// all pairs) against the page surface #050505.

export const SURFACE = "#050505";
export const INK = "#ededed";
export const INK2 = "#a3a3a0";
export const MUTED = "#6b6b68";

/** Express, Standard, Bulk (indexed by tier). */
export const TIER = ["#d95926", "#3987e5", "#199e70"] as const;
/** Robot 1 (and the single robot), 2, 3. Neutral white for R1, amber and pink for the fleet. */
export const ROBOT = ["#ededed", "#c98500", "#d55181"] as const;

export const FLOOR = "#0d0d0e";
export const SHELF_TOP = "#1b1b1e";
export const SHELF_FRONT = "#111113";
export const WALL = "#050505";

export function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
