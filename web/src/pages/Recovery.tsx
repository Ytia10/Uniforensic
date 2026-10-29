import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { runFile } from "../api";
import NeedRun from "../components/NeedRun";
import { Badge, Button, Card, DataKey, PageHeader, selectCls, Stat, StatRow, Td, Th } from "../components/ui";
import { useApp } from "../state";
import { axis, C, grid, REC_FILL, tooltip } from "../theme";

const TRUTH_TONE: Record<string, "good" | "bad" | "warn"> = { correct: "good", foreign: "bad", wrong: "warn" };
const TRUTH_WORD: Record<string, string> = { correct: "the owner's", foreign: "someone else's", wrong: "fabricated" };
const TABLE_NAME: Record<string, string> = { sms: "SMS", call: "calls", web: "web", login: "logins", txn: "payments" };

export default function Recovery() {
  const { run, reveal, go } = useApp();
  const [table, setTable] = useState("all");
  const rows = useMemo(() => (run?.recovered_records ?? []).filter((r) => table === "all" || r.table === table), [run, table]);
  if (!run) return <NeedRun />;
  const s = run.scenario;
  const fuzzyDrops = run.recovered_records.filter((r) => r.teammate_fuzzy_filter_drops).length;
  const tl = run.timeline.map((d) => ({ ...d, day: d.day.slice(5) }));

  return (
    <div className="space-y-6">
      <PageHeader step={2} title="Deleted-data recovery"
                  lead={<>The five databases were written by the real SQLite library, records were deleted on a timeline, and the carver read them back from the raw pages. This time the owner deleted <b>{s.behaviour === "immediate" ? "right after use" : "in a clean-up before seizure"}</b>{s.foreign && ", and someone else's deleted records were already on the phone"}.</>}
                  uses="Carving by N3, unchanged; confidence recalibrated by N5."
                  right={<Button onClick={() => go("triage")}>Next: triage</Button>} />
      <StatRow>
        <Stat label="records on the phone" value={s.n_artifacts} />
        <Stat label="deleted by the owner" value={s.n_deleted} note={reveal ? `${s.n_evidence_deleted} of the ${s.n_evidence} planted evidence records` : undefined} />
        <Stat label="recovered by N3" value={s.n_recovered} tone="rec" note="valid rows, not copies of live ones" />
        <Stat label="dropped by the N3's fuzzy filter" value={fuzzyDrops} tone={fuzzyDrops ? "warn" : "default"} note="recovered records it would discard" />
        {reveal ? (
          <Stat label="whose records they are" value={<span className="text-[1.3rem] leading-tight">{Object.entries(s.recovered_by_truth).map(([k, v]) => `${v} ${TRUTH_WORD[k] ?? k}`).join(", ") || "none"}</span>}
                note="from the benchmark's answer key" />
        ) : <Stat label="whose records they are" value="unknown" note="an examiner cannot tell; tick Show ground truth" />}
      </StatRow>

      <Card title="Activity per day"
            subtitle={reveal ? "Live records, the records the owner deleted, and the records N3 got back." : "Live records and the records N3 got back. What was deleted and lost stays invisible to an examiner."}
            right={<div className="flex flex-wrap gap-x-5 gap-y-1"><DataKey kind="live">live</DataKey>{reveal && <DataKey kind="deleted">deleted (truth)</DataKey>}<DataKey kind="recovered">recovered</DataKey></div>}>
        <div className="h-56">
          <ResponsiveContainer>
            <BarChart data={tl} margin={{ left: -18, right: 4 }} barGap={1}>
              <CartesianGrid vertical={false} {...grid} />
              <XAxis dataKey="day" {...axis} interval={3} />
              <YAxis {...axis} allowDecimals={false} />
              <Tooltip {...tooltip} />
              <Bar dataKey="live" name="live" fill={C.live} />
              {reveal && <Bar dataKey="deleted" name="deleted (truth)" fill="transparent" stroke={C.ink3} strokeDasharray="3 2" strokeWidth={1.25} />}
              <Bar dataKey="recovered" name="recovered" fill={REC_FILL} stroke={C.rec} strokeWidth={1} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Recovered records"
            subtitle="Each row was carved from deleted space. N3 p is the N3's score as shipped; confidence is N5's recalibrated score, in whole percent as CASE/UCO stores it."
            right={<select aria-label="Database" value={table} onChange={(e) => setTable(e.target.value)} className={`${selectCls} w-auto`}>
              <option value="all">All databases</option>{["sms", "call", "web", "login", "txn"].map((t) => <option key={t} value={t}>{TABLE_NAME[t]}</option>)}
            </select>}>
        <div className="max-h-[28rem] overflow-auto">
          <table className="w-full">
            <thead className="sticky top-0 z-10 bg-sheet"><tr><th className="w-2 border-b border-ink/70" aria-label="recovered" /><Th>Time</Th><Th>Database</Th><Th>Record</Th><Th right>N3 p</Th><Th right>Confidence</Th><Th>Found in</Th><Th>Fuzzy filter</Th>{reveal && <Th>Whose</Th>}</tr></thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td className="hatch-rec w-2 border-b border-sheet p-0" />
                  <Td mono className="whitespace-nowrap">{r.timestamp.replace("T", " ")}</Td><Td>{TABLE_NAME[r.table]}</Td><Td>{r.what}</Td>
                  <Td right>{r.p_raw.toFixed(3)}</Td><Td right>{r.confidence}%</Td>
                  <Td className="text-ink-2">{r.source_type}, {r.strategy}</Td>
                  <Td>{r.teammate_fuzzy_filter_drops ? <Badge tone="warn">would drop</Badge> : <span className="text-sm text-ink-3">keeps</span>}</Td>
                  {reveal && <Td><Badge tone={TRUTH_TONE[r.truth]}>{TRUTH_WORD[r.truth] ?? r.truth}{r.truth_role ? `, ${r.truth_role}` : ""}</Badge></Td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-2 text-sm text-ink-2">
          <span>The five databases, to open in any SQLite browser:</span>
          {s.databases.map((d) => <a key={d} className="font-mono text-[13px]" href={runFile(run.run_id, `databases/${d}`)}>{d}</a>)}
        </div>
      </Card>
    </div>
  );
}
