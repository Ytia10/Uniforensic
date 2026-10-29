import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api, follow, num, pct, runFile, type Job } from "../api";
import { Badge, Button, Card, ErrorBox, FigureImg, JobPanel, Loading, PageHeader, Stat, StatRow, Td, Th } from "../components/ui";
import { useResults } from "../components/useResults";
import { axis, C, grid, legend, REC_FILL, SERIES, tooltip } from "../theme";

const TABLES = ["sms", "call", "web", "login", "txn"];
const TABLE_NAME: Record<string, string> = { sms: "SMS", call: "calls", web: "web", login: "logins", txn: "payments" };
type Upload = { file: string; table: string; table_name: string; columns: string[]; live_rows: number;
                records: { values: unknown[]; valid_row: boolean; p_raw: number; confidence: number; strategy: string;
                           source_type: string; teammate_fuzzy_filter_drops: boolean | null }[] };

export default function N3Page() {
  const { data: d, error } = useResults("n3");
  const [pj, setPj] = useState<Job | null>(null);
  const [pr, setPr] = useState<{ report: Record<string, any>; plots: string[]; identical_to_n5: boolean } | null>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [uj, setUj] = useState<Job | null>(null);
  const [up, setUp] = useState<Upload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  if (error) return <ErrorBox error={error} />;
  if (!d) return <Loading />;
  const bench = d.teammate_benchmark, on1 = d.on_n1_databases, cal = on1.calibration;
  const methods = [["regex", "Regex baseline"], ["spec_only", "Spec-only (bring2lite style)"], ["n3", "N3"]].map(([k, n]) =>
    ({ name: n, recall: bench[k].recall, precision: bench[k].precision, live_fp: bench[k].fp_live }));
  const pragma = Object.entries(d.teammate_pragma).map(([k, v]) => ({ name: k, recall: (v as { recall: number }).recall }));
  const byTable = TABLES.map((t) => ({ table: TABLE_NAME[t], recovered: on1.recovery_rate_by_table[t], fuzzy: on1.fuzzy_filter_discard_rate_by_table[t] }));
  const relRaw = d.reliability.filter((r: { p: string }) => r.p === "raw");
  const relCal = d.reliability.filter((r: { p: string }) => r.p === "recalibrated");
  const pctTick = (v: number) => `${Math.round(v * 100)}%`;

  const runPipeline = async () => {
    setErr(null);
    try { const fin = await follow(await api.n3Pipeline(), setPj); if (fin.status === "done") setPr(await api.jobResult(fin.id)); }
    catch (e) { setErr((e as Error).message); }
  };
  const upload = async (f: File | undefined) => {
    if (!f) return;
    setErr(null); setUp(null);
    try { const fin = await follow(await api.n3Upload(f), setUj); if (fin.status === "done") setUp(await api.jobResult(fin.id)); }
    catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Ranking carved SQLite records with a calibrated confidence"
                  lead="A structure-first carver reads deleted rows from freeblocks, unallocated space and freelist pages. An explainable classifier with Platt scaling scores each candidate, and live rows are excluded. The code runs here exactly as shipped, checked by MD5."
                  uses="Novelty 3. Recalibration on N1's phones by N5." />
      <ErrorBox error={err} />
      <StatRow>
        <Stat label="recall and precision, their benchmark" value={`${pct(bench.n3.recall, 0)} / ${pct(bench.n3.precision, 0)}`} note={`live rows wrongly reported: ${bench.n3.fp_live}`} />
        <Stat label="reproduced bit for bit" value={d.equivalence.identical ? "yes" : "no"} tone={d.equivalence.identical ? "good" : "bad"} note={`numpy seed ${d.equivalence.numpy_seed}`} />
        <Stat label="calibration error on N1's phones" value={num(cal.ece_raw)} tone="warn" note={`ECE; on N3's data ${num(cal.teammate_in_sample_ece)}`} />
        <Stat label="after N5's recalibration" value={num(cal.ece_recalibrated)} tone="good" note="isotonic, per table, fitted on val_fit" />
      </StatRow>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="Their benchmark, on their own databases" subtitle="From the N3's run_pipeline() with the seed N5 uses.">
          <div className="h-60"><ResponsiveContainer><BarChart data={methods} margin={{ left: -14, right: 4 }} barGap={2}>
            <CartesianGrid vertical={false} {...grid} /><XAxis dataKey="name" {...axis} tick={{ ...axis.tick, fill: C.ink }} /><YAxis domain={[0, 1]} {...axis} tickFormatter={pctTick} />
            <Tooltip {...tooltip} formatter={(v) => pct(Number(v))} /><Legend {...legend} />
            <Bar dataKey="recall" name="recall" fill={SERIES[4]} /><Bar dataKey="precision" name="precision" fill={C.ink} />
          </BarChart></ResponsiveContainer></div>
          <p className="mt-3 text-sm text-ink-2">Their deletion-setting test, recall: {pragma.map((p) => `${p.name} ${pct(p.recall, 0)}`).join("; ")}.</p>
        </Card>
        <Card title="On N1's phones" subtitle="N5 step 5, test split. Share of deleted records recovered, and share of correctly recovered records the teammate's fuzzy live filter would wrongly drop.">
          <div className="h-60"><ResponsiveContainer><BarChart data={byTable} margin={{ left: -14, right: 4 }} barGap={2}>
            <CartesianGrid vertical={false} {...grid} /><XAxis dataKey="table" {...axis} tick={{ ...axis.tick, fill: C.ink }} /><YAxis domain={[0, 1]} {...axis} tickFormatter={pctTick} />
            <Tooltip {...tooltip} formatter={(v) => pct(Number(v))} /><Legend {...legend} />
            <Bar dataKey="recovered" name="recovered" fill={REC_FILL} stroke={C.rec} /><Bar dataKey="fuzzy" name="the fuzzy filter would drop" fill={SERIES[3]} />
          </BarChart></ResponsiveContainer></div>
        </Card>
      </div>
      <Card title="Does the confidence mean what it says?" subtitle="Each dot is a confidence bin on N1's phones; on the diagonal the confidence is honest. Dot size is the number of records.">
        <div className="h-72"><ResponsiveContainer><ScatterChart margin={{ left: -10, right: 12 }}>
          <CartesianGrid {...grid} /><XAxis type="number" dataKey="mean_p" domain={[0, 1]} name="confidence" {...axis} tickFormatter={pctTick}
                                             label={{ value: "confidence given", position: "insideBottomRight", offset: -2, fontSize: 12, fill: C.ink3 }} />
          <YAxis type="number" dataKey="share_genuine" domain={[0, 1]} name="share genuine" {...axis} tickFormatter={pctTick} /><ZAxis dataKey="count" range={[30, 400]} />
          <Tooltip {...tooltip} formatter={(v) => num(Number(v))} /><Legend {...legend} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={C.ink4} strokeDasharray="4 4" />
          <Scatter name="N3 confidence as shipped" data={relRaw} fill={SERIES[1]} /><Scatter name="after N5's recalibration" data={relCal} fill={C.ink} />
        </ScatterChart></ResponsiveContainer></div>
      </Card>

      <Card title="Live: run the N3's pipeline" subtitle="run_n3_pipeline.run_pipeline(), unchanged, in a scratch folder. About 10 to 30 seconds."
            right={<Button onClick={runPipeline} disabled={pj?.status === "running" || pj?.status === "queued"}>Run their pipeline</Button>}>
        {pj ? <JobPanel job={pj} showLog /> : <p className="text-[15px] text-ink-2">Shows N3's own console output, report and plots, then compares the classifier it trains with the one N5 uses.</p>}
        {pr && pj && (
          <div className="mt-5 space-y-4">
            <p className="flex flex-wrap items-center gap-2 text-[15px]">Classifier trained by N3's code, compared with the one N5 uses:
              <Badge tone={pr.identical_to_n5 ? "good" : "bad"}>{pr.identical_to_n5 ? "identical" : "different"}</Badge></p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              {pr.plots.map((p) => <FigureImg key={p} src={runFile(pj.id, `n3_run_output/plots/${p}`)} caption={`N3's plot: ${p}`} />)}
            </div>
          </div>
        )}
      </Card>

      <Card title="Live: carve a SQLite file" subtitle="Upload a database in one of the five workbench formats, for example one downloaded in step 2. N3 carves it and N5 recalibrates the confidence.">
        <label className="inline-flex cursor-pointer items-center gap-3 rounded-sm border border-rule bg-sheet px-3.5 py-[7px] text-[15px] font-semibold hover:border-ink-3">
          Choose a database file
          <input type="file" accept=".db,.sqlite,.sqlite3" onChange={(e) => upload(e.target.files?.[0])} className="sr-only" />
        </label>
        {uj && <div className="mt-4"><JobPanel job={uj} /></div>}
        {up && (
          <div className="mt-5">
            <p className="mb-3 text-[15px]"><span className="font-mono text-[13px]">{up.file}</span>: table <b>{up.table_name}</b>, {up.live_rows} live rows, <b className="text-rec">{up.records.filter((r) => r.valid_row).length} deleted rows recovered</b></p>
            <div className="max-h-80 overflow-auto">
              <table className="w-full">
                <thead className="sticky top-0 bg-sheet"><tr><th className="w-2 border-b border-ink/70" aria-label="recovered" />{up.columns.slice(1).map((c) => <Th key={c}>{c}</Th>)}<Th right>N3 p</Th><Th right>Confidence</Th><Th>Kind</Th></tr></thead>
                <tbody>{up.records.slice(0, 200).map((r, i) => (
                  <tr key={i} className={r.valid_row ? "" : "text-ink-3"}>
                    <td className={`w-2 border-b p-0 ${r.valid_row ? "hatch-rec border-sheet" : "border-rule-2"}`} />
                    {r.values.map((v, j) => <Td key={j}>{String(v).slice(0, 60)}</Td>)}
                    <Td right>{r.p_raw.toFixed(3)}</Td><Td right>{r.confidence}%</Td><Td>{r.valid_row ? "valid row" : "fragment"}</Td>
                  </tr>))}</tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
