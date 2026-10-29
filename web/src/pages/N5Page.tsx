import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { figure, num, pct } from "../api";
import { Card, ErrorBox, Field, FigureImg, Loading, PageHeader, selectCls, Stat, StatRow } from "../components/ui";
import { axis, C, grid, legend, REC_FILL, SERIES, tooltip } from "../theme";
import { useResults } from "../components/useResults";

const LEVELS = [0, 0.1, 0.25, 0.5, 1];
const VARS = ["L", "A", "Af", "B", "C"];
const COL: Record<string, string> = { L: SERIES[4], A: SERIES[1], Af: SERIES[3], B: SERIES[0], C: C.ink };
const W = (v: string) => (v === "C" ? 2.75 : 1.75);
type Row = Record<string, string | number>;

export default function N5Page() {
  const { data: d, error } = useResults("n5");
  const [rho, setRho] = useState("0.0");
  if (error) return <ErrorBox error={error} />;
  if (!d) return <Loading />;
  const f1 = d.macro_f1 as Row[], ter = d.ter as Row[], ben = d.false_incrimination as Row[];
  const byLevel = (rows: Row[], key: string) => LEVELS.map((l) => {
    const p: Row = { level: `${l * 100}%` };
    for (const v of VARS) { const r = rows.find((x) => x.variant === v && x.rho === rho && Number(x.level) === l); if (r) p[v] = Number(r[key]); }
    return p;
  });
  const fi = VARS.map((v) => ({
    variant: v,
    clean: Number(ben.find((r) => r.variant === v && r.rho === rho && r.foreign_rate === "clean")?.false_incrimination),
    residue: Number(ben.find((r) => r.variant === v && r.rho === rho && r.foreign_rate === "residue")?.false_incrimination),
  }));
  const work = LEVELS.map((l) => {
    const p: Row = { level: `${l * 100}%` };
    for (const v of ["A", "B", "C"]) {
      const r = (d.flag as Row[]).find((x) => x.variant === v && x.rho === rho && x.group === "offence_correct" && Number(x.level) === l);
      if (r) p[v] = Number(r.flag_rate);
    }
    return p;
  });
  const rates = [0, 0.1, 0.25, 0.5];
  const cov = rates.map((fr) => {
    const p: Row = { rate: `${fr * 100}%` };
    for (const v of ["A", "C"]) {
      const rs = (d.conformal as Row[]).filter((x) => x.variant === v && x.rho === rho && Number(x.foreign_rate) === fr && !["mixture", "benign_residue"].includes(String(x.level)));
      p[`${v} full`] = rs.reduce((a, x) => a + Number(x.coverage_full), 0) / rs.length;
      p[`${v} live`] = rs.reduce((a, x) => a + Number(x.coverage_live), 0) / rs.length;
    }
    return p;
  });
  const at = (rows: Row[], v: string, key: string, l = 1) => rows.find((x) => x.variant === v && x.rho === "0.0" && Number(x.level) === l)?.[key];

  return (
    <div className="space-y-6">
      <PageHeader title="Carrying recovery confidence and provenance into triage, explanations and reports"
                  lead="Five ways to feed carved records to the triage model, compared on the same phones: live only (L), merged (A), merged with the filter (Af), weighted by confidence (B) and two separate channels (C, the proposed design). Foreign residue tests what no confidence score can catch."
                  uses="Novelty 5. Figures from results_n5/ and figures_n5/ of your N5 run."
                  right={<div className="w-56"><Field label="Training residue prior">
                    <select value={rho} onChange={(e) => setRho(e.target.value)} className={selectCls}>
                      <option value="0.0">rho = 0</option><option value="0.25">rho = 0.25</option></select></Field></div>} />
      <StatRow>
        <Stat label="macro-F1 with all evidence deleted" value={`${num(Number(at(f1, "A", "mean")))} and ${num(Number(at(f1, "C", "mean")))}`} note="merged (A), then two channels (C); rho 0" />
        <Stat label="right class for the right reason" value={`${num(Number(at(ter, "A", "ter")))} and ${num(Number(at(ter, "C", "ter")))}`} note="TER, all evidence deleted; A, then C" />
        <Stat label="calibration error of N3's confidence" value={`${num(d.calibration.ece_raw)} to ${num(d.calibration.ece_recalibrated)}`} tone="good" note="ECE as shipped, then recalibrated" />
        <Stat label="CASE files with provenance facets" value={`${d.case_export.filter((c: { conforms: boolean }) => c.conforms).length} of ${d.case_export.length}`} tone="good" note="valid against CASE 1.5.0" />
      </StatRow>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="Triage accuracy as evidence is deleted" subtitle="Test macro-F1 against the share of planted evidence deleted. Clean benign phones, mean over three seeds.">
          <div className="h-64"><ResponsiveContainer><LineChart margin={{ left: -10, right: 8 }} data={byLevel(f1, "mean")}>
            <CartesianGrid {...grid} /><XAxis dataKey="level" {...axis} /><YAxis domain={[0, 1]} {...axis} />
            <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
            {VARS.map((v) => <Line key={v} dataKey={v} stroke={COL[v]} strokeWidth={W(v)} dot={{ r: 2.5 }} />)}
          </LineChart></ResponsiveContainer></div>
        </Card>
        <Card title="Right decision for the right reason" subtitle="TER: the class is right and Integrated Gradients finds at least two thirds of the planted evidence still visible.">
          <div className="h-64"><ResponsiveContainer><LineChart margin={{ left: -10, right: 8 }} data={byLevel(ter, "ter")}>
            <CartesianGrid {...grid} /><XAxis dataKey="level" {...axis} /><YAxis domain={[0, 1]} {...axis} />
            <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
            {VARS.map((v) => <Line key={v} dataKey={v} stroke={COL[v]} strokeWidth={W(v)} dot={{ r: 2.5 }} />)}
          </LineChart></ResponsiveContainer></div>
        </Card>
        <Card title="Innocent phones triaged as offences" subtitle="Clean benign phones, against benign phones holding someone else's deleted records.">
          <div className="h-64"><ResponsiveContainer><BarChart margin={{ left: -10, right: 8 }} barGap={2} data={fi}>
            <CartesianGrid vertical={false} {...grid} /><XAxis dataKey="variant" {...axis} tick={{ ...axis.tick, fontSize: 13, fill: C.ink }} /><YAxis domain={[0, 1]} {...axis} />
            <Tooltip {...tooltip} formatter={(v) => pct(Number(v))} /><Legend {...legend} />
            <Bar dataKey="clean" name="clean phone" fill={C.live} /><Bar dataKey="residue" name="with foreign residue" fill={REC_FILL} stroke={C.rec} />
          </BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Flag workload" subtitle="Share of correct offence decisions that rest on recovered records: the cases where the recovered data must be backed up.">
          <div className="h-64"><ResponsiveContainer><LineChart margin={{ left: -10, right: 8 }} data={work}>
            <CartesianGrid {...grid} /><XAxis dataKey="level" {...axis} /><YAxis domain={[0, 1]} {...axis} />
            <Tooltip {...tooltip} formatter={(v) => pct(Number(v))} /><Legend {...legend} />
            {["A", "B", "C"].map((v) => <Line key={v} dataKey={v} stroke={COL[v]} strokeWidth={W(v)} dot={{ r: 2.5 }} />)}
          </LineChart></ResponsiveContainer></div>
        </Card>
      </div>
      <Card title="Prediction-set coverage under foreign residue" subtitle="The target is 90% (dashed). The live-only set (dotted lines) keeps it; the set with recovered records (solid) does not.">
        <div className="h-64"><ResponsiveContainer><LineChart margin={{ left: -10, right: 8 }} data={cov}>
          <CartesianGrid {...grid} /><XAxis dataKey="rate" label={{ value: "benign phones with residue", position: "insideBottomRight", offset: -2, fontSize: 12, fill: C.ink3 }} />
          <YAxis domain={[0.75, 1]} ticks={[0.75, 0.8, 0.85, 0.9, 0.95, 1]} {...axis} /><Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
          <ReferenceLine y={0.9} stroke={C.ink3} strokeDasharray="5 4" />
          <Line dataKey="A full" name="A, with recovered" stroke={COL.A} strokeWidth={2} /><Line dataKey="A live" name="A, live only" stroke={COL.A} strokeWidth={2} strokeDasharray="2 3" />
          <Line dataKey="C full" name="C, with recovered" stroke={COL.C} strokeWidth={2.5} /><Line dataKey="C live" name="C, live only" stroke={COL.C} strokeWidth={2.5} strokeDasharray="2 3" />
        </LineChart></ResponsiveContainer></div>
      </Card>
      <section>
        <h2 className="mb-3 text-[1.2rem] font-semibold">Figures from the N5 run</h2>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          {d.figures.map((f: string) => <FigureImg key={f} src={figure("n5", f)} caption={`figures_n5/${f}`} />)}
        </div>
      </section>
    </div>
  );
}
