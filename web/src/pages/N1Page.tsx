import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { figure, num, pct } from "../api";
import { Card, ErrorBox, FigureImg, Loading, PageHeader, Stat, StatRow, Td, Th } from "../components/ui";
import { useResults } from "../components/useResults";
import { axis, C, grid, legend, SERIES, tooltip } from "../theme";

const EXPL = ["integrated_gradients", "kernelshap", "gradientshap", "lime", "feature_ablation", "random"];
const EXPL_NAME: Record<string, string> = {
  integrated_gradients: "Integrated Gradients", kernelshap: "KernelSHAP", gradientshap: "GradientSHAP",
  lime: "LIME", feature_ablation: "Feature ablation", random: "Random ranking (floor)",
};
const COLORS = [C.ink, SERIES[0], SERIES[2], SERIES[3], SERIES[1], C.rule];
const GROUP: Record<string, string> = {
  "all offence cases": "All offence cases", "oracle-passed": "Oracle passed: model right and uses the evidence", "oracle-failed": "Oracle failed",
  "tier=easy": "Tier: easy", "tier=mixed": "Tier: mixed", "tier=subtle": "Tier: subtle",
};

export default function N1Page() {
  const { data: d, error } = useResults("n1");
  if (error) return <ErrorBox error={error} />;
  if (!d) return <Loading />;
  const tiers = ["easy", "mixed", "subtle"];
  const rec = tiers.map((t) => {
    const row: Record<string, number | string> = { tier: t };
    for (const e of EXPL) {
      const r = d.recall_by_tier.find((x: { explainer: string; tier: string }) => x.explainer === e && x.tier === t);
      if (r) row[e] = r["recall@|G|"];
    }
    return row;
  });
  const l2 = d.acceptance.level2_test;
  return (
    <div className="space-y-6">
      <PageHeader title="Validated explanations on a planted-evidence benchmark"
                  lead="Evidence with a known answer key is planted in synthetic cases. Each explainer is scored on whether it finds that evidence (Level 1). For a single case, an explanation counts as fact only if two validated explainers agree and the model is confident (Level 2)."
                  uses="Novelty 1. All figures from results/ and figures/ of your N1 run." />
      <StatRow>
        <Stat label="triage macro-F1, test split" value={num(d.models.mlp.macro_f1)} note={`MLP; XGBoost reference ${num(d.models.xgb.macro_f1)}`} />
        <Stat label="acceptance threshold tau" value={d.acceptance.tau} note="calibrated on the validation split" />
        <Stat label="of validated explanations are right" value={pct(l2["all offence cases"].precision_of_validated)} tone="good"
              note={`${pct(l2["all offence cases"].validated_rate)} of offence cases get validated`} />
        <Stat label="of bad explanations are caught" value={pct(l2["all offence cases"].bad_caught_rate)} note="marked NOT VALIDATED" />
      </StatRow>
      <Card title="Level 1: does the explainer find the planted evidence?" subtitle="Strict recall@|G| per tier on the test split. 1.0 means every planted decisive feature is ranked on top.">
        <div className="h-72">
          <ResponsiveContainer>
            <BarChart data={rec} margin={{ left: -14, right: 4 }} barGap={2}>
              <CartesianGrid vertical={false} {...grid} />
              <XAxis dataKey="tier" {...axis} tick={{ ...axis.tick, fontSize: 13, fill: C.ink }} />
              <YAxis domain={[0, 1]} {...axis} />
              <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
              {EXPL.map((e, i) => <Bar key={e} dataKey={e} name={EXPL_NAME[e]} fill={COLORS[i]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card title="Level 2: the per-case acceptance gate" subtitle="Test split. Precision of validated = share of VALIDATED explanations that really point at the planted evidence.">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead><tr><Th>Group</Th><Th right>Cases</Th><Th right>Good explanations</Th><Th right>Validated</Th><Th right>Precision of validated</Th><Th right>Bad caught</Th></tr></thead>
            <tbody>{Object.entries(l2).map(([k, v], i) => {
              const r = v as Record<string, number | null>;
              return <tr key={k} className={i === 0 ? "font-semibold" : ""}><Td>{GROUP[k] ?? k}</Td><Td right>{r.n}</Td><Td right>{pct(r.share_good)}</Td><Td right>{pct(r.validated_rate)}</Td>
                <Td right>{pct(r.precision_of_validated)}</Td><Td right>{pct(r.bad_caught_rate)}</Td></tr>;
            })}</tbody>
          </table>
        </div>
      </Card>
      {d.scaffold && (
        <Card title="Scaffold attack: a model built to fool explainers" subtitle="Recall of the planted evidence before and after the attack, and how often an innocent feature is ranked first.">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr><Th>Explainer</Th><Th right>Recall before</Th><Th right>Recall after</Th><Th right>Innocent feature ranked first</Th></tr></thead>
              <tbody>{Object.entries(d.scaffold.explainers).map(([k, v]) => {
                const r = v as Record<string, number>;
                return <tr key={k}><Td>{EXPL_NAME[k] ?? k}</Td><Td right>{num(r.recall_before)}</Td><Td right>{num(r.recall_after)}</Td><Td right>{pct(r.innocent_feature_top1)}</Td></tr>;
              })}</tbody>
            </table>
          </div>
        </Card>
      )}
      <section>
        <h2 className="mb-3 text-[1.2rem] font-semibold">Figures from the N1 run</h2>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          {d.figures.map((f: string) => <FigureImg key={f} src={figure("n1", f)} caption={`figures/${f}`} />)}
        </div>
      </section>
    </div>
  );
}
