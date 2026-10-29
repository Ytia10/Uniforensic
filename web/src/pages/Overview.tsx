import { useEffect, useState } from "react";
import { api, num, pct, type Overview as O } from "../api";
import { Badge, Button, DataKey, ErrorBox, Loading } from "../components/ui";
import { useApp, type Page } from "../state";

const NOVELTY: Record<string, { name: string; question: string; page: Page }> = {
  N1: { name: "Validated explanations", question: "Can the examiner trust why the model decided?", page: "n1" },
  N2: { name: "Private explanation sharing", question: "Do shared explanation summaries leak an agency's cases?", page: "n2" },
  N3: { name: "SQLite carving", question: "Can deleted records be read back, and how sure is the tool?", page: "n3" },
  N4: { name: "Federated triage", question: "Can agencies train one model without pooling their cases?", page: "n4" },
  N5: { name: "Recovered-evidence provenance", question: "What happens when a decision rests on recovered records?", page: "n5" },
};

const FLOW: { title: string; what: string; page: Page }[] = [
  { title: "Seized phone", what: "five SQLite app databases, deleted records still in their free space", page: "intake" },
  { title: "Carve", what: "N3 reads deleted rows back from the raw pages", page: "recovery" },
  { title: "Calibrate and tag", what: "N5 recalibrates the confidence and marks each record as recovered", page: "recovery" },
  { title: "Triage", what: "the model reads live and recovered records as separate inputs", page: "triage" },
  { title: "Explain and check", what: "Integrated Gradients, then N1's acceptance gate", page: "triage" },
  { title: "State the certainty", what: "the recovery flag and two 90% prediction sets", page: "certainty" },
  { title: "Exchange and report", what: "a CASE/UCO file with provenance, and a printable report", page: "exchange" },
];

export default function Overview() {
  const { go } = useApp();
  const [o, setO] = useState<O | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = (refresh = false) => {
    setBusy(true);
    api.overview(refresh).then(setO).catch((e) => setErr(String(e.message))).finally(() => setBusy(false));
  };
  useEffect(() => load(), []);

  const items = o ? Object.values(o.integrity.novelties) : [];
  const ok = items.reduce((a, i) => a + i.ok, 0), checked = items.reduce((a, i) => a + i.checked, 0);
  const allOk = items.length > 0 && items.every((i) => i.status === "ok");

  return (
    <div className="space-y-10">
      <section className="grid grid-cols-1 gap-x-12 gap-y-8 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="max-w-[44rem]">
          <h1 className="text-[2.6rem] font-semibold leading-[1.05] text-ink">A federated and explainable framework for digital forensics</h1>
          <p className="mt-4 text-[1.0625rem] leading-relaxed text-ink-2">
            Analyse a seized phone from deleted data to a signed report, using all five parts of the framework, then open the
            experiment behind each part. Every figure here is read from your own N1-N5 results or computed live by their code.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3">
            <Button onClick={() => go("intake")}>Analyse a phone</Button>
            <div className="flex flex-wrap gap-x-5 gap-y-2">
              <DataKey kind="live">live record</DataKey>
              <DataKey kind="deleted">deleted record</DataKey>
              <DataKey kind="recovered">recovered from deleted space</DataKey>
            </div>
          </div>
        </div>

        <div className="self-start rounded-md border border-rule bg-sheet px-6 py-5">
          <h2 className="text-[1.15rem] font-semibold text-ink">Integrity of the results</h2>
          {!o ? <Loading what="Checking the N1-N5 result files" /> : (
            <>
              <div className={`mt-3 font-display text-[2.6rem] font-semibold leading-none num ${allOk ? "text-good" : "text-bad"}`}>
                {ok}<span className="text-ink-4"> / </span>{checked}
              </div>
              <p className="mt-2 text-[15px] leading-normal text-ink">
                {allOk ? "result files match the fingerprints recorded when each experiment ran."
                  : `result files match. ${checked - ok} differ from, or are missing against, what their experiment recorded.`}
              </p>
              <p className="mt-2 text-sm text-ink-3">SHA-256 or MD5 per file, checked {o.integrity.checked_at}.</p>
              <div className="mt-4"><Button kind="secondary" onClick={() => load(true)} disabled={busy}>{busy ? "Checking..." : "Check again"}</Button></div>
            </>
          )}
        </div>
      </section>

      <ErrorBox error={err} />
      {o && Object.keys(o.missing).length > 0 && <ErrorBox error={`Missing outputs: ${Object.values(o.missing).join(" ")}`} />}

      {o && (
        <section aria-labelledby="contrib">
          <h2 id="contrib" className="mb-3 text-[1.35rem] font-semibold text-ink">Five contributions, one headline result each</h2>
          <div className="rounded-md border border-rule bg-sheet">
            {o.headlines.map((h, i) => {
              const n = NOVELTY[h.novelty], it = o.integrity.novelties[h.novelty];
              return (
                <div key={h.novelty} className={`grid grid-cols-1 gap-x-8 gap-y-3 px-6 py-5 md:grid-cols-[3rem_minmax(0,1.3fr)_minmax(0,1fr)_10rem] md:items-baseline ${i ? "border-t border-rule-2" : ""}`}>
                  <div className="font-display text-2xl font-semibold text-ink-3">{h.novelty}</div>
                  <div>
                    <button onClick={() => go(n.page)} className="text-left text-[1.0625rem] font-semibold text-ink underline decoration-ink-4 underline-offset-[3px] hover:decoration-ink">{n.name}</button>
                    <p className="mt-1 text-[15px] text-ink-2">{n.question}</p>
                  </div>
                  <div>
                    <div className="font-display text-[1.9rem] font-semibold leading-none text-ink num">{h.fmt === "pct" ? pct(h.value) : num(h.value)}</div>
                    <div className="mt-1.5 text-sm font-semibold text-ink">{h.title}</div>
                    <div className="text-[13px] leading-snug text-ink-3">{h.note}</div>
                  </div>
                  <div className="text-sm text-ink-2 md:text-right">
                    {!it ? "-" : it.status === "ok"
                      ? <><span className="num">{it.ok} of {it.checked}</span> files unchanged</>
                      : <span title={[...it.changed, ...it.missing].join(", ")}><Badge tone="bad">{it.status}</Badge> {[...it.changed, ...it.missing].slice(0, 2).join(", ")}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section aria-labelledby="flow">
        <h2 id="flow" className="text-[1.35rem] font-semibold text-ink">How one phone moves through the framework</h2>
        <p className="mt-1 max-w-[44rem] text-[15px] text-ink-2">Steps 1 to 6 in the menu run this chain live on the case you choose; one phone takes about two seconds.</p>
        <ol className="mt-5 grid grid-cols-1 gap-px overflow-hidden rounded-md border border-rule bg-rule sm:grid-cols-2 xl:grid-cols-7">
          {FLOW.map((f, i) => (
            <li key={f.title} className="bg-sheet">
              <button onClick={() => go(f.page)} className="block h-full w-full px-4 py-4 text-left hover:bg-well">
                <div className="font-display text-lg font-semibold text-ink-3 num">{i + 1}</div>
                <div className="mt-1 text-[15px] font-semibold leading-snug text-ink">{f.title}</div>
                <div className="mt-1 text-sm leading-snug text-ink-2">{f.what}</div>
              </button>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
