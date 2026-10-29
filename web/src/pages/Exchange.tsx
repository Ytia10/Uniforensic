import { useEffect, useState } from "react";
import { api, follow, runFile, type Job } from "../api";
import NeedRun from "../components/NeedRun";
import { Button, Card, ErrorBox, JobPanel, PageHeader, Stat, StatRow } from "../components/ui";
import { useApp } from "../state";

type Node = { "@id": string; "@type": string; "core:hasFacet": { "@type": string }[] };

export default function Exchange() {
  const { run, go } = useApp();
  const [sample, setSample] = useState<{ recovered: Node | null; live: Node | null; counts: Record<string, number> } | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [verdict, setVerdict] = useState<{ conforms: boolean; report: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!run) return;
    setSample(null); setVerdict(null); setJob(null);
    fetch(runFile(run.run_id, "case.jsonld")).then((r) => r.json()).then((doc: { "@graph": Node[] }) => {
      const g = doc["@graph"];
      const hasRec = (n: Node) => n["core:hasFacet"]?.some((f) => f["@type"] === "observable:RecoveredObjectFacet");
      const counts: Record<string, number> = {};
      for (const n of g) counts[n["@type"]] = (counts[n["@type"]] ?? 0) + 1;
      counts["recovered objects"] = g.filter(hasRec).length;
      setSample({ recovered: g.find(hasRec) ?? null,
                  live: g.find((n) => n["@type"] === "observable:Message" && !hasRec(n)) ?? null, counts });
    }).catch((e) => setErr(String(e)));
  }, [run]);

  if (!run) return <NeedRun />;
  const validate = async () => {
    setErr(null);
    try {
      const fin = await follow(await api.validate(run.run_id), setJob, 800);
      if (fin.status === "done") setVerdict(await api.jobResult(fin.id));
    } catch (e) { setErr((e as Error).message); }
  };
  const running = job?.status === "running" || job?.status === "queued";

  return (
    <div className="space-y-6">
      <PageHeader step={5} title="Evidence exchange"
                  lead="Another agency must be able to rebuild the same decision from the file it receives. Every record is exported in CASE/UCO, the open standard for digital evidence; recovered records also carry UCO's recovered-status and confidence facets."
                  uses="CASE/UCO mapping from N4; provenance facets from N5."
                  right={<Button onClick={() => go("report")}>Next: report</Button>} />
      <ErrorBox error={err} />
      <StatRow>
        <Stat label="objects in the CASE file" value={run.case_export.objects} />
        <Stat label="recovered objects" value={sample?.counts["recovered objects"] ?? "-"} tone="rec" note="each with recovery status and confidence" />
        <Stat label="SHA-256 of the file" value={<span className="font-mono text-[1.05rem] font-medium">{run.case_export.sha256.slice(0, 16)}...</span>} note="the same file for the same case and scenario" />
        <Stat label="official validator" value={verdict ? (verdict.conforms ? "conforms" : "fails") : running ? "running" : "not run yet"}
              tone={verdict ? (verdict.conforms ? "good" : "bad") : "default"} note="case_utils 0.18.0 against CASE 1.5.0" />
      </StatRow>

      <Card title="Validate and download"
            subtitle="The validator checks the file against the CASE 1.5.0 ontology and its SHACL shapes."
            right={<div className="flex flex-wrap gap-2">
              <a href={runFile(run.run_id, "case.jsonld")} className="no-underline"><Button kind="secondary">Download the .jsonld file</Button></a>
              <Button onClick={validate} disabled={running}>Run the CASE validator (about 1-1.5 min)</Button>
            </div>}>
        {job ? <JobPanel job={job} /> : <p className="text-[15px] text-ink-2">The validation runs here, offline; nothing is sent anywhere.</p>}
        {verdict && !verdict.conforms && <pre className="mt-3 max-h-64 overflow-auto rounded-xs bg-ink p-3 font-mono text-xs text-white">{verdict.report}</pre>}
      </Card>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card title="A recovered record, as exported" subtitle="Its two provenance facets come first, then the whole object: N4's facet, then observable:RecoveredObjectFacet and core:ConfidenceFacet.">
          {sample?.recovered && (
            <pre className="mb-3 overflow-auto rounded-xs border border-rec/40 bg-rec-tint p-3 font-mono text-xs leading-5 text-ink">{JSON.stringify(sample.recovered["core:hasFacet"].filter((f) => f["@type"] !== sample.recovered!["core:hasFacet"][0]["@type"]), null, 2)}</pre>
          )}
          <pre className="max-h-80 overflow-auto rounded-xs bg-ink p-3 font-mono text-xs leading-5 text-white">{sample?.recovered ? JSON.stringify(sample.recovered, null, 2) : "This case has no recovered record."}</pre>
        </Card>
        <Card title="A live record, as exported" subtitle="The same N4 mapping, with no provenance facets.">
          <pre className="max-h-96 overflow-auto rounded-xs bg-ink p-3 font-mono text-xs leading-5 text-white">{sample?.live ? JSON.stringify(sample.live, null, 2) : "-"}</pre>
        </Card>
      </div>
    </div>
  );
}
