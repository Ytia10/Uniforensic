import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, follow, type CaseDetail, type CaseRow, type Job, type ModelInfo, type RunResult } from "../api";
import { Badge, Button, Card, checkCls, ClassTag, ErrorBox, Field, JobPanel, Loading, PageHeader, selectCls, Td, Th } from "../components/ui";
import { useApp } from "../state";
import { axis, grid, legend, SERIES, tooltip } from "../theme";

const TABLES = ["sms", "call", "web", "login", "txn"] as const;
const TABLE_NAME: Record<string, string> = { sms: "SMS", call: "calls", web: "web", login: "logins", txn: "payments" };
const VARIANT_NAME: Record<string, string> = {
  L: "L: live records only", A: "A: recovered records merged in (N3 p at least 0.5)", Af: "Af: merged, with the live filter",
  B: "B: recovered records weighted by confidence", C: "C: two channels, live and recovered",
};

export default function Intake() {
  const { reveal, setRun, go } = useApp();
  const [cases, setCases] = useState<CaseRow[] | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [levels, setLevels] = useState<number[]>([]);
  const [tier, setTier] = useState("all");
  const [label, setLabel] = useState("all");
  const [q, setQ] = useState("");
  const [pageNo, setPageNo] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [tableFilter, setTableFilter] = useState("all");
  const [level, setLevel] = useState(1.0);
  const [behaviour, setBehaviour] = useState("seeded");
  const [foreign, setForeign] = useState(false);
  const [model, setModel] = useState({ variant: "C", rho: 0.25, seed: 2026 });
  const [kernelshap, setKernelshap] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [recent, setRecent] = useState<Awaited<ReturnType<typeof api.runs>>["runs"]>([]);

  useEffect(() => {
    api.cases().then((r) => setCases(r.cases)).catch((e) => setErr(e.message));
    api.models().then((m) => { setModels(m.models); setLevels(m.levels); setModel(m.default); }).catch((e) => setErr(e.message));
    api.runs().then((r) => setRecent(r.runs)).catch(() => undefined);
  }, []);
  useEffect(() => {
    setDetail(null);
    if (sel) api.caseDetail(sel).then(setDetail).catch((e) => setErr(e.message));
  }, [sel]);

  const filtered = useMemo(() => (cases ?? []).filter((c) =>
    (tier === "all" || c.tier === tier) && (!reveal || label === "all" || c.label === label) &&
    c.case_id.toLowerCase().includes(q.toLowerCase())), [cases, tier, label, q, reveal]);
  const PER = 12;
  const pageRows = filtered.slice(pageNo * PER, pageNo * PER + PER);
  const selCase = cases?.find((c) => c.case_id === sel);
  const inGrid = behaviour === "seeded" && (!foreign || selCase?.label === "benign");

  const daily = useMemo(() => {
    if (!detail) return [];
    const m = new Map<string, Record<string, number>>();
    for (const a of detail.artifacts) {
      const d = a.timestamp.slice(0, 10);
      const row = m.get(d) ?? { sms: 0, call: 0, web: 0, login: 0, txn: 0 };
      row[a.table] += 1;
      m.set(d, row);
    }
    return [...m.entries()].sort().map(([day, r]) => ({ day: day.slice(5), ...r }));
  }, [detail]);

  const run = async () => {
    if (!sel) return;
    setErr(null);
    try {
      const j = await api.analyse({ case_id: sel, level, foreign, behaviour, ...model, kernelshap });
      const fin = await follow(j, setJob);
      if (fin.status === "done") {
        const r = await api.jobResult<RunResult>(fin.id);
        setRun(r);
        api.runs().then((x) => setRecent(x.runs)).catch(() => undefined);
      }
    } catch (e) { setErr((e as Error).message); }
  };

  const openRun = async (id: string) => {
    try { setRun(await api.run(id)); go("recovery"); } catch (e) { setErr((e as Error).message); }
  };

  return (
    <div className="space-y-6">
      <PageHeader step={1} title="Case intake"
                  lead="Choose one of the 600 held-out test cases, then decide what its owner deleted before the phone was seized. The workbench writes the phone's five SQLite databases and analyses them while you watch."
                  uses="Cases from the N1 benchmark; deletion scenario and models from N5." />
      <ErrorBox error={err} />
      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Card title="Test cases" subtitle={reveal ? "Ground truth is shown: you can filter by class." : "The class stays hidden until the analysis, as for an examiner."}>
          <div className="mb-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
            <input value={q} onChange={(e) => { setQ(e.target.value); setPageNo(0); }} placeholder="Search case id" aria-label="Search case id" className={selectCls} />
            <select aria-label="Tier" value={tier} onChange={(e) => { setTier(e.target.value); setPageNo(0); }} className={selectCls}>
              <option value="all">All tiers</option><option value="easy">Easy</option><option value="mixed">Mixed</option><option value="subtle">Subtle</option>
            </select>
            <select aria-label="Class" value={label} disabled={!reveal} onChange={(e) => { setLabel(e.target.value); setPageNo(0); }} className={selectCls}>
              <option value="all">All classes</option>
              {["benign", "fraud", "harassment", "cybercrime"].map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          {!cases ? <Loading what="Loading cases" /> : (
            <>
              <table className="w-full">
                <thead><tr><Th>Case</Th><Th>Tier</Th>{reveal && <Th>Class</Th>}<Th right>Records</Th></tr></thead>
                <tbody>
                  {pageRows.map((c) => (
                    <tr key={c.case_id} onClick={() => setSel(c.case_id)} aria-selected={sel === c.case_id}
                        className={`cursor-pointer ${sel === c.case_id ? "bg-well font-semibold [&>td:first-child]:shadow-[inset_3px_0_0_var(--color-ink)]" : "hover:bg-well"}`}>
                      <Td mono>{c.case_id}</Td><Td>{c.tier}</Td>
                      {reveal && <Td><ClassTag cls={c.label} /></Td>}
                      <Td right>{c.n_artifacts}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4 flex items-center justify-between text-sm text-ink-3">
                <span className="num">{filtered.length} {filtered.length === 1 ? "case" : "cases"}, page {pageNo + 1} of {Math.max(1, Math.ceil(filtered.length / PER))}</span>
                <div className="flex gap-2">
                  <Button kind="secondary" disabled={pageNo === 0} onClick={() => setPageNo(pageNo - 1)}>Previous</Button>
                  <Button kind="secondary" disabled={(pageNo + 1) * PER >= filtered.length} onClick={() => setPageNo(pageNo + 1)}>Next</Button>
                </div>
              </div>
            </>
          )}
        </Card>

        <div className="space-y-6">
          {!sel ? (
            <Card title="Earlier analyses on this machine" subtitle="Reopen one, or choose a case on the left to start a new one.">
              {recent.length === 0 ? <p className="text-[15px] text-ink-2">Nothing analysed yet.</p> : (
                <table className="w-full">
                  <thead><tr><Th>When</Th><Th>Version</Th><Th>Model</Th><Th>Triage</Th><Th /></tr></thead>
                  <tbody>{recent.slice(0, 8).map((r) => (
                    <tr key={r.run_id}><Td className="num whitespace-nowrap">{r.when}</Td><Td mono>{r.version_id}</Td>
                      <Td>{r.model.variant}, rho {r.model.rho}, seed {r.model.seed}</Td>
                      <Td><span className="flex flex-wrap items-center gap-2"><ClassTag cls={r.pred} />{r.flag && <Badge tone="accent">rests on recovered</Badge>}</span></Td>
                      <Td right><Button kind="quiet" onClick={() => openRun(r.run_id)}>Open</Button></Td></tr>))}
                  </tbody>
                </table>
              )}
            </Card>
          ) : !detail ? <Card><Loading what={`Loading ${sel}`} /></Card> : (
            <>
              <Card title={<>Case <span className="font-mono text-[1.05rem] font-medium">{detail.case_id}</span></>}
                    subtitle={`${detail.artifacts.length} records on the phone before anything was deleted. Tier: ${detail.tier}.`}
                    right={reveal ? <ClassTag cls={detail.label} prefix="truth: " /> : undefined}>
                <div className="h-48">
                  <ResponsiveContainer>
                    <BarChart data={daily} margin={{ left: -18, right: 4 }}>
                      <CartesianGrid vertical={false} {...grid} />
                      <XAxis dataKey="day" {...axis} interval={3} />
                      <YAxis {...axis} allowDecimals={false} />
                      <Tooltip {...tooltip} />
                      <Legend {...legend} />
                      {TABLES.map((t, i) => <Bar key={t} dataKey={t} name={TABLE_NAME[t]} stackId="a" fill={SERIES[i]} />)}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 flex items-center justify-between gap-4">
                  <span className="text-sm text-ink-3">Records per day by database{reveal && ". Planted evidence is tinted red, distractors amber."}</span>
                  <select aria-label="Database" value={tableFilter} onChange={(e) => setTableFilter(e.target.value)} className={`${selectCls} w-auto`}>
                    <option value="all">All databases</option>{TABLES.map((t) => <option key={t} value={t}>{TABLE_NAME[t]}</option>)}
                  </select>
                </div>
                <div className="mt-3 max-h-56 overflow-auto border-t border-rule-2">
                  <table className="w-full">
                    <tbody>
                      {detail.artifacts.filter((a) => tableFilter === "all" || a.table === tableFilter).slice(0, 300).map((a) => (
                        <tr key={a.artifact_id} className={reveal && a.role === "evidence" ? "bg-bad-tint" : reveal && a.role === "distractor" ? "bg-warn-tint" : ""}>
                          <Td mono className="whitespace-nowrap">{a.timestamp.replace("T", " ")}</Td><Td>{TABLE_NAME[a.table]}</Td><Td>{a.what}</Td>
                          {reveal && <Td>{a.role !== "background" && <Badge tone={a.role === "evidence" ? "bad" : "warn"}>{a.role} {a.pattern}</Badge>}</Td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>

              <Card title="Deletion scenario and model" subtitle="What the owner did before seizure, and which N5 model triages the phone.">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field label="Share of planted evidence deleted">
                    <select value={level} onChange={(e) => setLevel(Number(e.target.value))} className={selectCls}>
                      {levels.map((l) => <option key={l} value={l}>{Math.round(l * 100)}%</option>)}
                    </select>
                  </Field>
                  <Field label="When the owner deleted it">
                    <select value={behaviour} onChange={(e) => setBehaviour(e.target.value)} className={selectCls}>
                      <option value="seeded">As N5 drew it for this case</option>
                      <option value="immediate">Right after use (within 72 h)</option>
                      <option value="clean-up">In a clean-up before seizure (last 48 h)</option>
                    </select>
                  </Field>
                  <div className="sm:col-span-2"><Field label="Model">
                    <select value={`${model.variant}|${model.rho}|${model.seed}`} className={selectCls}
                            onChange={(e) => { const [v, r, s] = e.target.value.split("|"); setModel({ variant: v, rho: Number(r), seed: Number(s) }); }}>
                      {models.map((m) => (
                        <option key={`${m.variant}|${m.rho}|${m.seed}`} value={`${m.variant}|${m.rho}|${m.seed}`}>
                          {VARIANT_NAME[m.variant]}; rho {m.rho}, seed {m.seed}
                        </option>))}
                    </select>
                  </Field></div>
                  <div className="flex flex-col gap-3 sm:col-span-2">
                    <label className="flex items-start gap-3 text-[15px]"><input type="checkbox" checked={foreign} onChange={(e) => setForeign(e.target.checked)} className={`${checkCls} mt-0.5`} />
                      <span><span className="font-semibold">Foreign residue on the device</span><span className="block text-sm text-ink-3">Someone else's deleted records are still in the free space: a shared or second-hand phone.</span></span></label>
                    <label className="flex items-start gap-3 text-[15px]"><input type="checkbox" checked={kernelshap} onChange={(e) => setKernelshap(e.target.checked)} className={`${checkCls} mt-0.5`} />
                      <span><span className="font-semibold">Run N1's acceptance check</span><span className="block text-sm text-ink-3">Adds KernelSHAP so two explainers can be compared.</span></span></label>
                  </div>
                </div>
                <div className="mt-5 flex flex-wrap items-end justify-between gap-4 border-t border-rule-2 pt-4">
                  <p className="max-w-[34rem] text-sm leading-normal text-ink-2">
                    {inGrid ? "N5 evaluated this scenario, so the live result will be compared with the stored N5 run." :
                      "N5 did not evaluate this scenario (a forced deletion time, or residue on an offence case). It still runs live; there is just no stored run to compare with."}
                    {selCase?.label === "benign" && reveal && " A benign case has no planted evidence, so the deleted share does not change its phone."}
                  </p>
                  <Button onClick={run} disabled={job?.status === "running" || job?.status === "queued"}>Analyse this phone</Button>
                </div>
              </Card>
              {job && (
                <Card title="Analysis" subtitle="Each line is one part of the N1-N5 code, run on this phone."
                      right={job.status === "done" ? <Button onClick={() => go("recovery")}>Next: recovery</Button> : undefined}>
                  <JobPanel job={job} />
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
