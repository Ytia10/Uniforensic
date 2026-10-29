import { useEffect, useState, type ReactNode } from "react";
import { api, consistencyText, pct, type Overview } from "../api";
import NeedRun from "../components/NeedRun";
import { Button } from "../components/ui";
import { useApp } from "../state";

const VARIANT = { L: "live data only", A: "recovered records merged with live data", Af: "merged, live filter",
                  B: "recovered records weighted by confidence", C: "separate live and recovered channels" } as Record<string, string>;

function H({ n, children }: { n: number; children: ReactNode }) {
  return <h2 className="mt-8 flex items-baseline gap-3 text-[1.3rem] font-semibold text-ink"><span className="text-ink-4 num">{n}</span>{children}</h2>;
}

function Row({ k, v }: { k: string; v: ReactNode }) {
  return <tr><th scope="row" className="w-60 border-b border-rule-2 py-2 pr-5 text-left align-top text-sm font-semibold text-ink-2">{k}</th><td className="border-b border-rule-2 py-2 text-[15px] text-ink">{v}</td></tr>;
}

export default function Report() {
  const { run, reveal } = useApp();
  const [ov, setOv] = useState<Overview | null>(null);
  useEffect(() => { api.overview().then(setOv).catch(() => undefined); }, []);
  if (!run) return <NeedRun />;
  const t = run.triage, e = run.explanation, s = run.scenario;
  const byTable = run.recovered_records.reduce<Record<string, number>>((m, r) => ({ ...m, [r.table]: (m[r.table] ?? 0) + 1 }), {});
  const confs = run.recovered_records.map((r) => r.confidence);
  const now = new Date().toISOString().slice(0, 16).replace("T", " ");

  return (
    <div className="mx-auto max-w-[52rem] space-y-4">
      <div className="no-print flex items-center justify-between gap-4"><p className="text-sm text-ink-3">Step 6 of 6. Only the report itself is printed.</p><Button onClick={() => window.print()}>Print / save as PDF</Button></div>
      <article className="print-area rounded-md border border-rule bg-sheet px-12 py-10">
        <header className="border-b-2 border-ink pb-5">
          <div className="text-sm text-ink-3">UniForensic Workbench, automated triage report</div>
          <h1 className="mt-1 text-[2rem] font-semibold leading-tight">Case {run.case_id} <span className="font-mono text-base font-normal text-ink-3">version {run.version_id}</span></h1>
          <div className="mt-2 text-sm text-ink-2">Generated {now}, run <span className="font-mono text-[13px]">{run.run_id}</span>. Decision support only: an examiner must confirm every finding below.</div>
        </header>

        <H n={1}>Finding</H>
        <p className="mt-2 text-[1.0625rem] leading-relaxed">
          The device is triaged as <b>{t.pred}</b> (probability {pct(t.confidence)}).
          {" "}The 90% prediction set is <b>{`{${run.sets.with_recovered.join(", ")}}`}</b> with recovered records and
          <b> {`{${run.sets.live_only.join(", ")}}`}</b> from live data only.
        </p>
        {run.flag.depends_on_recovered ? (
          <p className="mt-4 flex gap-4 rounded-sm border border-rec/40 bg-rec-tint p-4 text-[15px] leading-relaxed"><span className="hatch-rec mt-1 inline-block h-8 w-8 shrink-0 rounded-xs" aria-hidden /><span>
            <b>This conclusion depends on recovered records.</b> Without them the model says {t.pred_without_recovered}. Their confidence
            ({Math.min(...confs)}-{Math.max(...confs)}%) says the records are intact, not who created them: confirm attribution (who used
            the device, and when) before relying on them.</span>
          </p>
        ) : <p className="mt-3 text-[15px] text-ink-2">The conclusion does not depend on recovered records: live data alone gives the same class.</p>}

        <H n={2}>Basis of the finding</H>
        <table className="mt-2 w-full"><tbody>
          <Row k="Explanation (Integrated Gradients)" v={<ol className="list-decimal pl-5">{e.top_inputs.slice(0, 5).map((x) => <li key={x.name}>{x.label} <span className="text-ink-3">(value {x.value.toFixed(2)}, {x.attribution >= 0 ? "+" : ""}{x.attribution.toFixed(3)})</span></li>)}</ol>} />
          <Row k="Explanation status (N1 Level 2)" v={e.acceptance.validated ? "VALIDATED: the two validated explainers agree and the model is confident"
            : `NOT VALIDATED: ${e.acceptance.reasons.join("; ")}. The explanation is shown as an indication, not as a fact.`} />
          <Row k="Share of attribution on recovered records" v={run.flag.recovered_attribution_share == null ? "n/a (model without a recovered channel)" : pct(run.flag.recovered_attribution_share, 0)} />
        </tbody></table>

        <H n={3}>Recovered data</H>
        <table className="mt-2 w-full"><tbody>
          <Row k="Records on the device" v={`${s.n_artifacts} artifacts in 5 SQLite databases (messages, calls, browser_history, logins, transactions)`} />
          <Row k="Recovered from deleted space" v={`${s.n_recovered} records: ${Object.entries(byTable).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}`} />
          <Row k="Recovery tool" v="SQLiteCarver + CalibratedSQLiteClassifier (unchanged), confidence recalibrated per table on N1 databases (N5)" />
          <Row k="Calibrated confidence" v={confs.length ? `${Math.min(...confs)}-${Math.max(...confs)}% (median ${[...confs].sort((a, b) => a - b)[Math.floor(confs.length / 2)]}%)` : "-"} />
        </tbody></table>

        <H n={4}>Method and reproducibility</H>
        <table className="mt-2 w-full"><tbody>
          <Row k="Triage model" v={`N5 variant ${run.model.variant} (${VARIANT[run.model.variant]}), trained with residue prior rho = ${run.model.rho}, seed ${run.model.seed}; N1's MLP recipe`} />
          <Row k="Prediction sets" v={`split conformal (LAC), alpha = ${run.sets.alpha}, calibrated on N1 validation cases never used for training`} />
          <Row k="Evidence exchange" v={`CASE/UCO 1.5.0 file, ${run.case_export.objects} objects, SHA-256 ${run.case_export.sha256}`} />
          <Row k="Consistency with the stored experiment" v={consistencyText(run, reveal)} />
          <Row k="Integrity of the framework outputs" v={ov ? Object.entries(ov.integrity.novelties).map(([k, v]) => `${k} ${v.status} (${v.ok}/${v.checked})`).join("; ") : "..."} />
        </tbody></table>

        {reveal && (
          <>
            <h2 className="mt-8 text-[1.15rem] font-semibold text-bad">Benchmark answer key, not part of a real report</h2>
            <p className="mt-2 text-[15px]">Truth: <b>{run.truth.label}</b> ({run.truth.tier}); planted evidence {run.truth.evidence_patterns.join(", ") || "none"};
              still visible after deletion: {run.truth.visible_features.join(", ") || "none"}; recovered records by origin:
              {" "}{Object.entries(s.recovered_by_truth).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}.</p>
          </>
        )}
        <footer className="mt-10 border-t border-rule pt-3 text-sm leading-normal text-ink-3">
          Limitations: simulated devices with real SQLite files (journal mode DELETE, secure_delete OFF); confidence measures record integrity, not attribution;
          coverage guarantees hold for devices like the calibration devices. Generated by the UniForensic Workbench from the N1-N5 code and outputs on this machine.
        </footer>
      </article>
    </div>
  );
}
