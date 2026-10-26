// Colour tokens. Tier and robot sets are validated (dataviz validate_palette.js, dark mode,
// all pairs) against the page surface #0b0d12.

export const SURFACE = "#0b0d12";
export const INK = "#f0efec";
export const INK2 = "#c3c2b7";
export const MUTED = "#898781";

/** Express, Standard, Bulk (indexed by tier). */
export const TIER = ["#d95926", "#3987e5", "#199e70"] as const;
/** Robot 1, 2, 3. */
export const ROBOT = ["#9085e9", "#c98500", "#d55181"] as const;

export const FLOOR = "#10131a";
export const SHELF_TOP = "#222838";
export const SHELF_FRONT = "#161a26";
export const WALL = "#07080c";

export function rgba(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
