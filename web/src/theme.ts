// Colours and chart settings shared by every page. Keep in step with index.css.
import { createElement } from "react";

export const C = {
  bench: "#eef1f0", sheet: "#ffffff", well: "#f5f7f6",
  ink: "#17201d", ink2: "#45514c", ink3: "#5f6b66", ink4: "#9aa5a0",
  rule: "#d3dad7", rule2: "#e5eae8",
  live: "#3f4b47", rec: "#7a2e6e", recTint: "#f5eaf2",
  good: "#2b6a49", warn: "#835500", bad: "#a3352b",
};

/** Fill for anything recovered from deleted space (pattern defined once in App.tsx). */
export const REC_FILL = "url(#hatch-rec)";

/** Muted series colours for comparisons that are not about live vs recovered. */
export const SERIES = ["#3b6a99", "#b3612b", "#4c8569", "#c29a2a", "#6f7c77", "#17201d"];

export const CLASS_COLOR: Record<string, string> = {
  benign: "#6f7c77", fraud: "#b3612b", harassment: "#3b6a99", cybercrime: "#c29a2a",
};

const FONT = '"Source Sans 3 Variable", "Segoe UI", sans-serif';

export const axis = {
  tick: { fontSize: 12, fill: C.ink3, fontFamily: FONT },
  stroke: C.rule,
  tickLine: false,
} as const;

export const grid = { stroke: C.rule2, strokeDasharray: "0" } as const;

export const tooltip = {
  contentStyle: { fontFamily: FONT, fontSize: 13, border: `1px solid ${C.rule}`, borderRadius: 3, boxShadow: "none", color: C.ink },
  labelStyle: { color: C.ink, fontWeight: 600 },
  cursor: { fill: "rgba(23,32,29,0.04)" },
} as const;

export const legend = {
  wrapperStyle: { fontFamily: FONT, fontSize: 13, paddingTop: 6 },
  iconSize: 11,
  // legend words stay in text colour; only the key carries the series colour
  formatter: (value: string) => createElement("span", { style: { color: C.ink2 } }, value),
} as const;
