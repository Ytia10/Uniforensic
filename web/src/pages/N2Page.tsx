import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, figure, follow, num, type Job } from "../api";
import { Button, Card, ErrorBox, Field, FigureImg, JobPanel, Loading, PageHeader, selectCls, Stat, StatRow } from "../components/ui";
import { useResults } from "../components/useResults";
import { axis, C, grid, legend, REC_FILL, SERIES, tooltip } from "../theme";

type Rel = { scheme: string; eps: number | null; auc: number; tpr_1pct: number; evidence_recall3: number };
const EPS = ["0.5", "1", "2", "4", "8", "inf"];

export default function N2Page() {
  const { data: d, error } = useResults("n2");
  const [job, setJob] = useState<Job | null>(null);
  const [ready, setReady] = useState(false);
  const [eps, setEps] = useState("1");
  const [rep, setRep] = useState(0);
  const [rel, setRel] = useState<Awaited<ReturnType<typeof api.privacyRelease>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  if (error) return <ErrorBox error={error} />;
  if (!d) return <Loading />;

  const rows: Rel[] = d.release;
  const ref = (s: string) => rows.find((r) => r.scheme === s);
  const curve = EPS.filter((e) => e !== "inf").map((e) => {
    const p: Record<string, number | string> = { eps: e };
    for (const s of ["dp", "secagg_dp"]) {
      const r = rows.find((x) => x.scheme === s && x.eps === Number(e));
      if (r) { p[`${s} AUC`] = r.auc; p[`${s} recall`] = r.evidence_recall3; }
    }
    return p;
  });

  const prepare = async () => {
    setErr(null);
    try { const fin = await follow(await api.privacyPrepare(), setJob); if (fin.status === "done") { setReady(true); await release(eps, rep); } }
    catch (e) { setErr((e as Error).message); }
  };
  const release = async (e: string, r: number) => {
    try { setRel(await api.privacyRelease(e, r)); } catch (x) { setErr((x as Error).message); }
  };
  const lines = (k: "AUC" | "recall") => <>
    <Line dataKey={`dp ${k}`} name="differential privacy" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} />
    <Line dataKey={`secagg_dp ${k}`} name="secure aggregation + differential privacy" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} />
  </>;

  return (
    <div className="space-y-6">
      <PageHeader title="What shared explanation summaries leak, and what protecting them costs"
                  lead="Agencies in the federation share per-class explanation summaries. A tracing attacker tries to tell whether a case was in an agency's data. Differential privacy (Gaussian noise) and secure aggregation reduce that, at a cost in how faithful the summary stays."
                  uses="Novelty 2. Figures from results_n2/ and figures_n2/ of your N2 run." />
      <StatRow>
        <Stat label="attack AUC, no protection" value={num(ref("none")?.auc)} note="0.5 means the attacker is guessing" />
        <Stat label="attack AUC, secure aggregation" value={num(ref("secagg")?.auc)} />
        <Stat label="evidence recall@3 of the summary" value={num(ref("secagg")?.evidence_recall3)} note="with secure aggregation: it still points at the evidence" />
        <Stat label="schemes and epsilons evaluated" value={rows.length} />
      </StatRow>
      <Card title="Privacy against faithfulness, by epsilon" subtitle="Left: lower AUC means less leakage. Right: higher recall means the summary still points at the planted evidence. Means over seeds and noise draws.">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <figure>
            <figcaption className="mb-1 text-sm font-semibold">Tracing attack AUC</figcaption>
            <div className="h-64"><ResponsiveContainer><LineChart data={curve} margin={{ left: -10, right: 8 }}>
              <CartesianGrid {...grid} /><XAxis dataKey="eps" {...axis} label={{ value: "epsilon", position: "insideBottomRight", offset: -2, fontSize: 12, fill: C.ink3 }} />
              <YAxis domain={[0.45, 0.6]} {...axis} /><Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
              <ReferenceLine y={ref("none")?.auc} stroke={C.bad} strokeDasharray="4 4" label={{ value: "no protection", fontSize: 12, fill: C.bad, position: "insideTopRight" }} />
              <ReferenceLine y={0.5} stroke={C.ink4} strokeDasharray="2 3" label={{ value: "guessing", fontSize: 12, fill: C.ink3, position: "insideBottomRight" }} />
              {lines("AUC")}
            </LineChart></ResponsiveContainer></div>
          </figure>
          <figure>
            <figcaption className="mb-1 text-sm font-semibold">Evidence recall@3 of the released summary</figcaption>
            <div className="h-64"><ResponsiveContainer><LineChart data={curve} margin={{ left: -10, right: 8 }}>
              <CartesianGrid {...grid} /><XAxis dataKey="eps" {...axis} /><YAxis domain={[0, 1]} {...axis} />
              <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
              {lines("recall")}
            </LineChart></ResponsiveContainer></div>
          </figure>
        </div>
      </Card>

      <Card title="Live: publish how much decisions rest on recovered records"
            subtitle="N2's release mechanism applied to N5's federated two-channel model. Per class, the share of decisive attribution that sits on recovered records, released with noise."
            right={!ready ? <Button onClick={prepare} disabled={job?.status === "running"}>Prepare (about 20-40 s)</Button> : undefined}>
        <ErrorBox error={err} />
        {job && !ready && <JobPanel job={job} />}
        {!job && !ready && <p className="text-[15px] text-ink-2">Prepare loads the federated model and computes the exact shares once; every release after that is instant.</p>}
        {ready && (
          <>
            <div className="mb-5 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:w-2/3">
              <Field label="Epsilon (privacy budget)"><select className={selectCls} value={eps} onChange={(e) => { setEps(e.target.value); release(e.target.value, rep); }}>
                {EPS.map((e) => <option key={e} value={e}>{e === "inf" ? "no noise" : e}</option>)}</select></Field>
              <Field label={`Noise draw ${rep + 1} of 20`}><input type="range" min={0} max={19} value={rep} className="mt-2 w-full accent-[#17201d]"
                     onChange={(e) => { setRep(Number(e.target.value)); release(eps, Number(e.target.value)); }} /></Field>
            </div>
            {rel && (
              <div className="h-60"><ResponsiveContainer><BarChart data={rel.rows} margin={{ left: -14, right: 4 }} barGap={2}>
                <CartesianGrid vertical={false} {...grid} /><XAxis dataKey="class" {...axis} tick={{ ...axis.tick, fontSize: 13, fill: C.ink }} />
                <YAxis domain={[0, 1]} {...axis} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
                <Bar dataKey="exact_reliance" name="exact share, no privacy" fill={C.rule} />
                <Bar dataKey="released_reliance" name="released with differential privacy" fill={REC_FILL} stroke={C.rec} />
              </BarChart></ResponsiveContainer></div>
            )}
            {rel && <p className="mt-2 text-sm text-ink-2">Mean absolute error at this epsilon and draw: <b className="text-ink num">{num(rel.rows.reduce((a, r) => a + r.abs_error, 0) / rel.rows.length, 4)}</b>. Each draw is seeded exactly as N5 step 13 seeds it, so these are the numbers in <span className="font-mono text-[13px]">results_n5/dp_release.csv</span>.</p>}
          </>
        )}
      </Card>
      <section>
        <h2 className="mb-3 text-[1.2rem] font-semibold">Figures from the N2 run</h2>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
          {d.figures.map((f: string) => <FigureImg key={f} src={figure("n2", f)} caption={`figures_n2/${f}`} />)}
        </div>
      </section>
    </div>
  );
}
