import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, figure, follow, num, type Job } from "../api";
import { Badge, Button, Card, ErrorBox, Field, FigureImg, JobPanel, Loading, PageHeader, selectCls, Stat, StatRow, Td, Th } from "../components/ui";
import { axis, C, grid, legend, SERIES, tooltip } from "../theme";
import { useResults } from "../components/useResults";

const ALPHAS = [0.1, 0.5, 1, 100];
type FedOut = { identical_to_stored: boolean | null; stored_model: string | null; history: number[]; attacker: number | null;
                test?: Record<string, number> };

export default function N4Page() {
  const { data: d, error } = useResults("n4");
  const [form, setForm] = useState({ alpha: 0.5, seed: 2026, method: "fedavg", attack: "none", rule: "fedavg" });
  const [job, setJob] = useState<Job | null>(null);
  const [out, setOut] = useState<FedOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pAlpha, setPAlpha] = useState(0.1);
  const live = useMemo(() => (job?.lines ?? []).map((l) => /round\s+(\d+): validation macro-F1 ([\d.]+)/.exec(l))
    .filter(Boolean).map((m) => ({ round: Number(m![1]), f1: Number(m![2]) })), [job]);
  if (error) return <ErrorBox error={error} />;
  if (!d) return <Loading />;

  const centralF1 = d.central.find((r: { index: string }) => r.index === "macro_f1");
  const byAlpha = ALPHAS.map((a) => {
    const f = (m: string) => d.federated.find((r: { method: string; alpha: number }) => r.method === m && r.alpha === a);
    const l = d.local.find((r: { alpha: number }) => r.alpha === a);
    return { alpha: String(a), FedAvg: f("fedavg")?.macro_f1_mean, FedProx: f("fedprox")?.macro_f1_mean, "Local only": l?.macro_f1_mean };
  });
  const conv = Array.from({ length: (d.convergence["fedavg|0.5"] ?? []).length }, (_, i) => {
    const p: Record<string, number> = { round: i + 1 };
    for (const a of ALPHAS) { const h = d.convergence[`fedavg|${a}`]; if (h) p[`alpha ${a}`] = h[i]; }
    return p;
  });
  const parts = d.partitions.filter((r: { alpha: number; seed: number }) => r.alpha === pAlpha && r.seed === 2026);
  const poisoned = form.attack !== "none";

  const start = async () => {
    setErr(null); setOut(null);
    try { const fin = await follow(await api.fedRun(form), setJob, 500); if (fin.status === "done") setOut(await api.jobResult(fin.id)); }
    catch (e) { setErr((e as Error).message); }
  };
  const set = (k: string, v: string | number) => setForm((f) => {
    const n = { ...f, [k]: v };
    if (k === "attack") { n.method = "fedavg"; n.alpha = v === "none" ? n.alpha : 0.5; n.rule = v === "none" ? "fedavg" : n.rule; }
    if (k === "method" && n.attack === "none") n.rule = String(v);
    return n;
  });

  return (
    <div className="space-y-6">
      <PageHeader title="Federated triage across agencies on one shared schema"
                  lead="Eight simulated agencies hold skewed case mixes; a small Dirichlet alpha means very skewed. Every agency computes the same 27 features from CASE/UCO records, so together they train one model without sharing a case. Robust aggregation counters one malicious agency."
                  uses="Novelty 4. Figures from results_n4/ and figures_n4/ of your N4 run." />
      <StatRow>
        <Stat label="macro-F1, all cases in one place" value={num(centralF1?.mean)} note="centralised: the upper bound" />
        <Stat label="macro-F1 with FedAvg, alpha 0.5" value={num(byAlpha[1].FedAvg)} note="eight agencies, no case shared" />
        <Stat label="macro-F1 of one agency alone, alpha 0.1" value={num(byAlpha[0]["Local only"])} tone="warn" note="mean over agencies" />
        <Stat label="CASE/UCO round trip" value={d.case_export.feature_mismatches?.length === 0 ? "exact" : "check"} note={`${d.case_export.cases} cases, ${d.case_export.objects} objects`} />
      </StatRow>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="Accuracy by case-mix skew" subtitle="Test macro-F1, mean over three seeds. The dashed line is the centralised model.">
          <div className="h-64"><ResponsiveContainer><LineChart data={byAlpha} margin={{ left: -10, right: 8 }}>
            <CartesianGrid {...grid} /><XAxis dataKey="alpha" {...axis} label={{ value: "alpha", position: "insideBottomRight", offset: -2, fontSize: 12, fill: C.ink3 }} />
            <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => Number(v).toFixed(2)} /><Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
            <ReferenceLine y={centralF1?.mean} stroke={C.ink3} strokeDasharray="4 4" label={{ value: "centralised", fontSize: 12, fill: C.ink3, position: "insideBottomRight" }} />
            <Line dataKey="FedAvg" stroke={C.ink} strokeWidth={2.5} dot={{ r: 3 }} /><Line dataKey="FedProx" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} /><Line dataKey="Local only" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} />
          </LineChart></ResponsiveContainer></div>
        </Card>
        <Card title="Convergence with FedAvg" subtitle="Validation macro-F1 after each round, mean over seeds.">
          <div className="h-64"><ResponsiveContainer><LineChart data={conv} margin={{ left: -10, right: 8 }}>
            <CartesianGrid {...grid} /><XAxis dataKey="round" {...axis} /><YAxis domain={[0, 1]} {...axis} />
            <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
            {ALPHAS.map((a, i) => <Line key={a} dataKey={`alpha ${a}`} dot={false} strokeWidth={2} stroke={[SERIES[1], C.ink, SERIES[0], SERIES[4]][i]} />)}
          </LineChart></ResponsiveContainer></div>
        </Card>
      </div>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="Poisoning: one malicious agency" subtitle="Test macro-F1, fraud F1 and the rate of right decisions for the right reason (TER), by attack and aggregation rule. Means over seeds.">
          <div className="overflow-x-auto"><table className="w-full"><thead><tr><Th>Attack</Th><Th>Rule</Th><Th right>Macro-F1</Th><Th right>Fraud F1</Th><Th right>TER</Th></tr></thead>
            <tbody>{d.poisoning.map((r: Record<string, string | number>) => (
              <tr key={`${r.attack}${r.rule}`}><Td>{r.attack}</Td><Td>{r.rule}</Td><Td right>{num(Number(r.macro_f1))}</Td><Td right>{num(Number(r.f1_fraud))}</Td><Td right>{num(Number(r.ter_integrated_gradients))}</Td></tr>))}
            </tbody></table></div>
        </Card>
        <Card title="Who holds which cases" subtitle="Training cases per agency and class, seed 2026."
              right={<select aria-label="alpha" value={pAlpha} onChange={(e) => setPAlpha(Number(e.target.value))} className={`${selectCls} w-auto`}>
                {ALPHAS.map((a) => <option key={a} value={a}>alpha {a}</option>)}</select>}>
          <div className="overflow-x-auto"><table className="w-full"><thead><tr><Th>Agency</Th><Th right>benign</Th><Th right>fraud</Th><Th right>harassment</Th><Th right>cybercrime</Th><Th right>total</Th></tr></thead>
            <tbody>{parts.map((r: Record<string, number>) => (
              <tr key={r.agency}><Td>{r.agency}</Td>{["benign", "fraud", "harassment", "cybercrime", "total"].map((c) => <Td key={c} right>{r[c]}</Td>)}</tr>))}
            </tbody></table></div>
        </Card>
      </div>

      <Card title="Live: train the federation" subtitle="N4's own round loop with Flower's aggregation code. The trained model is compared with the one N4 stored for the same setting.">
        <ErrorBox error={err} />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-6">
          <Field label="alpha"><select className={selectCls} value={form.alpha} disabled={poisoned} onChange={(e) => set("alpha", Number(e.target.value))}>
            {ALPHAS.map((a) => <option key={a} value={a}>{a}</option>)}</select></Field>
          <Field label="seed"><select className={selectCls} value={form.seed} onChange={(e) => set("seed", Number(e.target.value))}>
            {[2026, 2027, 2028].map((s) => <option key={s}>{s}</option>)}</select></Field>
          <Field label="method"><select className={selectCls} value={form.method} disabled={poisoned} onChange={(e) => set("method", e.target.value)}>
            <option value="fedavg">FedAvg</option><option value="fedprox">FedProx</option></select></Field>
          <Field label="attack"><select className={selectCls} value={form.attack} onChange={(e) => set("attack", e.target.value)}>
            <option value="none">none</option><option value="label_flip">label flip (fraud to benign)</option><option value="scaled_update">scaled update (x-5)</option></select></Field>
          <Field label="aggregation"><select className={selectCls} value={form.rule} disabled={!poisoned} onChange={(e) => set("rule", e.target.value)}>
            {(poisoned ? ["fedavg", "fedmedian", "fedtrimmedavg", "krum"] : [form.method]).map((r) => <option key={r}>{r}</option>)}</select></Field>
          <div className="flex items-end"><Button onClick={start} disabled={job?.status === "running" || job?.status === "queued"}>Train (about 10-40 s)</Button></div>
        </div>
        {job && (
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <JobPanel job={job} />
            <div className="h-64"><ResponsiveContainer><LineChart data={live} margin={{ left: -10, right: 8 }}>
              <CartesianGrid {...grid} /><XAxis dataKey="round" {...axis} /><YAxis domain={[0, 1]} {...axis} />
              <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Line dataKey="f1" name="validation macro-F1" stroke={C.ink} strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart></ResponsiveContainer></div>
          </div>
        )}
        {out && (
          <div className="mt-4 flex flex-wrap items-center gap-3 text-[15px]">
            <Badge tone={out.identical_to_stored ? "good" : out.identical_to_stored === null ? "neutral" : "bad"}>
              {out.identical_to_stored ? `identical to ${out.stored_model}` : out.identical_to_stored === null ? "no stored model" : "differs from stored model"}</Badge>
            {out.test && <span className="num">test macro-F1 <b>{num(out.test.macro_f1)}</b>, fraud F1 {num(out.test.f1_fraud)}, TER {num(out.test.ter_integrated_gradients)}</span>}
            {out.attacker != null && <span className="text-ink-3">malicious agency: {out.attacker}</span>}
          </div>
        )}
      </Card>
      <section>
        <h2 className="mb-3 text-[1.2rem] font-semibold">Figures from the N4 run</h2>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          {d.figures.map((f: string) => <FigureImg key={f} src={figure("n4", f)} caption={`figures_n4/${f}`} />)}
        </div>
      </section>
    </div>
  );
}
