import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import NeedRun from "../components/NeedRun";
import { Badge, Button, Card, DataKey, PageHeader, Td, Th } from "../components/ui";
import { useApp } from "../state";
import { pct } from "../api";
import { axis, C, CLASS_COLOR, grid, REC_FILL, tooltip } from "../theme";

const fmt = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(2));

export default function Triage() {
  const { run, reveal, go } = useApp();
  if (!run) return <NeedRun />;
  const t = run.triage, e = run.explanation, a = e.acceptance;
  const probs = t.classes.map((c, i) => ({ cls: c, with: t.proba[i], without: t.proba_without_recovered[i] }));
  const attr = [...e.top_inputs].reverse().map((x) => ({ ...x, short: x.label.length > 50 ? x.label.slice(0, 48) + "..." : x.label }));
  const flips = t.pred !== t.pred_without_recovered;

  return (
    <div className="space-y-6">
      <PageHeader step={3} title="Triage and explanation"
                  lead="The model's decision, the inputs that drove it, and whether N1's acceptance gate allows the explanation to be reported as fact."
                  uses={`Model: N5 variant ${run.model.variant}, rho ${run.model.rho}, seed ${run.model.seed}. Explanation check: N1.`}
                  right={<Button onClick={() => go("certainty")}>Next: certainty</Button>} />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Card title="Decision" subtitle="Probability of each class, with and without the recovered records.">
          <div className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <span className="font-display text-[2.4rem] font-semibold leading-none num">{pct(t.confidence)}</span>
            <span className="inline-flex items-center gap-2 font-display text-[1.75rem] font-semibold leading-none">
              <span className="inline-block h-3.5 w-3.5 rounded-[1px]" style={{ background: CLASS_COLOR[t.pred] }} aria-hidden />{t.pred}
            </span>
            {reveal && <span className="text-sm text-ink-3">truth: <b className="text-ink">{run.truth.label}</b>, {run.truth.tier}</span>}
          </div>
          <div className="mb-2 flex flex-wrap gap-x-5 gap-y-1"><DataKey kind="recovered">with recovered records</DataKey><DataKey kind="live">live records only</DataKey></div>
          <div className="h-52">
            <ResponsiveContainer>
              <BarChart data={probs} margin={{ left: -18, right: 4 }} barGap={2}>
                <CartesianGrid vertical={false} {...grid} />
                <XAxis dataKey="cls" {...axis} />
                <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip {...tooltip} formatter={(v) => pct(Number(v))} />
                <Bar dataKey="with" name="with recovered records" fill={REC_FILL} stroke={C.rec} strokeWidth={1} />
                <Bar dataKey="without" name="live records only" fill={C.live} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className={`mt-3 text-[15px] ${flips ? "font-semibold text-ink" : "text-ink-2"}`}>
            Without the recovered records the same model says {t.pred_without_recovered}{flips ? ", a different class." : ", the same class."}
          </p>
        </Card>

        <Card title="Why: the inputs that pushed the decision" subtitle="Integrated Gradients, the explainer N1 validated. A bar to the right pushes towards the predicted class."
              right={<div className="flex flex-wrap gap-x-5 gap-y-1"><DataKey kind="recovered">recovered channel</DataKey><DataKey kind="live">live</DataKey></div>}>
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={attr} layout="vertical" margin={{ left: 0, right: 16 }}>
                <CartesianGrid horizontal={false} {...grid} />
                <XAxis type="number" {...axis} />
                <YAxis type="category" dataKey="short" width={300} {...axis} tick={{ ...axis.tick, fill: C.ink2, fontSize: 12.5 }} />
                <ReferenceLine x={0} stroke={C.ink3} />
                <Tooltip {...tooltip} formatter={(v) => Number(v).toFixed(4)} labelFormatter={(l) => String(l)} />
                <Bar dataKey="attribution" name="attribution">
                  {attr.map((x, i) => <Cell key={i} fill={x.channel === "recovered" ? REC_FILL : C.live} stroke={x.channel === "recovered" ? C.rec : "none"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Can the explanation be reported as fact?"
            subtitle="N1's Level-2 gate: only if two validated explainers agree and the model is confident enough. Otherwise the report says NOT VALIDATED and why."
            right={<Badge tone={a.validated ? "good" : "warn"}>{a.validated ? "VALIDATED" : "NOT VALIDATED"}</Badge>}>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-[1fr_1fr_1.4fr]">
          <div>
            <div className="font-display text-[1.9rem] font-semibold leading-none num">{a.agreement_top5 == null ? "not run" : pct(a.agreement_top5, 0)}</div>
            <div className="mt-2 text-sm font-semibold">KernelSHAP and IG agree on the top five</div>
            <div className="text-[13px] text-ink-3">needs at least {pct(a.min_agreement, 0)}</div>
          </div>
          <div>
            <div className="font-display text-[1.9rem] font-semibold leading-none num">{pct(t.confidence)}</div>
            <div className="mt-2 text-sm font-semibold">model confidence</div>
            <div className="text-[13px] text-ink-3">needs tau = {a.tau}, calibrated by N1 on N1's own model</div>
          </div>
          <div className="text-[15px]">
            <div className="mb-1 text-sm font-semibold">Result</div>
            {a.reasons.length === 0 ? <p className="text-good">All criteria are met.</p> :
              <ul className="list-disc space-y-1 pl-5 text-ink">{a.reasons.map((r) => <li key={r}>{r.charAt(0).toUpperCase() + r.slice(1)}.</li>)}</ul>}
          </div>
        </div>
      </Card>

      <Card title="The nine decisive features" subtitle="The value each N5 variant sees for the features that define the offences. Rows where the recovered channel carries something are marked.">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead><tr><th className="w-2 border-b border-ink/70" aria-label="recovered" /><Th>Feature</Th><Th right>Live only (L)</Th><Th right>Merged (A)</Th><Th right>Weighted (B)</Th><Th right>Recovered channel (C)</Th>{reveal && <Th>Planted evidence</Th>}</tr></thead>
            <tbody>{run.decisive_features.map((f) => (
              <tr key={f.name}>
                <td className={`w-2 border-b p-0 ${f.recovered_channel > 0 ? "hatch-rec border-sheet" : "border-rule-2"}`} />
                <Td>{f.label}</Td><Td right>{fmt(f.live)}</Td><Td right>{fmt(f.merged_A)}</Td><Td right>{fmt(f.weighted_B)}</Td>
                <Td right className={f.recovered_channel > 0 ? "font-semibold text-rec" : ""}>{fmt(f.recovered_channel)}</Td>
                {reveal && <Td>{f.planted ? (f.visible ? <Badge tone="bad">planted, visible</Badge> : <Badge>planted, lost</Badge>) : ""}</Td>}
              </tr>))}
            </tbody>
          </table>
        </div>
        {reveal && (
          <p className="mt-4 text-sm text-ink-2">
            Answer-key check, as in N1: of the explanation's top {e.top_base_features.length} features,{" "}
            {e.top_base_features.filter((x) => x.planted_and_visible).length} are planted evidence still visible in this version
            ({run.truth.visible_features.length} visible of {run.truth.planted_features.length} planted).
          </p>
        )}
      </Card>
    </div>
  );
}
