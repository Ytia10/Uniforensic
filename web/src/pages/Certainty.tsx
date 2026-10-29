import NeedRun from "../components/NeedRun";
import { Badge, Button, Card, ClassTag, DataKey, PageHeader } from "../components/ui";
import { consistencyText, pct } from "../api";
import { useApp } from "../state";

function SetChips({ classes, all, label }: { classes: string[]; all: string[]; label: string }) {
  return (
    <ul className="flex flex-wrap gap-2" aria-label={label}>
      {all.map((c) => {
        const inSet = classes.includes(c);
        return (
          <li key={c} className={`rounded-sm border px-3 py-1.5 text-[15px] ${inSet ? "border-ink bg-ink font-semibold text-white" : "border-rule text-ink-4 line-through"}`}>
            {c}<span className="sr-only">{inSet ? " (in the set)" : " (ruled out)"}</span>
          </li>
        );
      })}
    </ul>
  );
}

export default function Certainty() {
  const { run, reveal, go } = useApp();
  if (!run) return <NeedRun />;
  const f = run.flag, s = run.sets, c = run.consistency;
  const nRec = run.scenario.n_recovered;
  const confs = run.recovered_records.map((r) => r.confidence);
  const truthIn = (set: string[]) => set.includes(run.truth.label);

  return (
    <div className="space-y-6">
      <PageHeader step={4} title="Provenance and certainty"
                  lead="Carving confidence says a recovered record is intact, never whose it is. So the report must say when a decision rests on recovered records, and what the evidence still allows without them."
                  uses="Recovery-dependence flag and conformal prediction sets from N5."
                  right={<Button onClick={() => go("exchange")}>Next: evidence exchange</Button>} />

      <section className={`flex gap-5 rounded-md border px-6 py-6 ${f.depends_on_recovered ? "border-rec/40 bg-rec-tint" : "border-good/40 bg-good-tint"}`}>
        <span className={`mt-1 hidden h-12 w-12 shrink-0 rounded-sm sm:block ${f.depends_on_recovered ? "hatch-rec" : "solid-live"}`} aria-hidden />
        <div className="max-w-[52rem]">
          <h2 className="text-[1.75rem] font-semibold leading-tight text-ink">
            {f.depends_on_recovered ? "This conclusion depends on recovered records." : "This conclusion stands on live records alone."}
          </h2>
          <p className="mt-2 text-base leading-relaxed text-ink">
            Run again with all {nRec} recovered records removed, the same model says <b>{run.triage.pred_without_recovered}</b>
            {f.depends_on_recovered ? <> instead of <b>{run.triage.pred}</b>.</> : ", the same class."}
            {f.depends_on_recovered && ` Their calibrated confidence (${Math.min(...confs)} to ${Math.max(...confs)}%) says they are intact, not who created them. Confirm who used the device, and when, before relying on them.`}
          </p>
          {f.recovered_attribution_share != null && (
            <p className="mt-2 text-[15px] text-ink-2">The explanation puts <b className="text-ink">{pct(f.recovered_attribution_share, 0)}</b> of the decisive-feature attribution on the recovered channel.
              The explanation-based flag fires at 50%, so it {f.recovered_attribution_share >= 0.5 ? "fires too" : "does not fire"}.</p>
          )}
        </div>
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="90% prediction set with the recovered records"
              subtitle={`The classes the evidence cannot rule out. Split conformal, LAC, q = ${s.q_full.toFixed(3)}.`}
              right={reveal ? <Badge tone={truthIn(s.with_recovered) ? "good" : "bad"}>{truthIn(s.with_recovered) ? "contains the truth" : "misses the truth"}</Badge> : <DataKey kind="recovered">uses recovered</DataKey>}>
          <SetChips classes={s.with_recovered} all={run.triage.classes} label="prediction set with recovered records" />
          <p className="mt-4 text-sm leading-normal text-ink-2">It contains the true class 90% of the time when new phones resemble the calibration phones. Foreign residue breaks that assumption.</p>
        </Card>
        <Card title="90% prediction set from live records only"
              subtitle={`The same model with the recovered records removed. q = ${s.q_live.toFixed(3)}.`}
              right={reveal ? <Badge tone={truthIn(s.live_only) ? "good" : "bad"}>{truthIn(s.live_only) ? "contains the truth" : "misses the truth"}</Badge> : <DataKey kind="live">live only</DataKey>}>
          <SetChips classes={s.live_only} all={run.triage.classes} label="prediction set from live records" />
          <p className="mt-4 text-sm leading-normal text-ink-2">Foreign residue can only enter through recovered records, so this set keeps its 90% guarantee whatever someone else left on the phone. The difference between the two sets is what the recovered records add.</p>
        </Card>
      </div>

      <Card title="Is this the stored experiment?" subtitle="The live result compared with N5's stored run of the same phone."
            right={c.in_n5_grid ? <Badge tone={c.identical ? "good" : "bad"}>{c.identical ? "matches the stored run" : "differs"}</Badge> : <Badge>not in N5's grid</Badge>}>
        <p className="text-[15px] text-ink">{consistencyText(run, reveal)}</p>
        {reveal && (
          <p className="mt-2 text-sm text-ink-2">Truth: {run.truth.label}. Recovered records by origin: {Object.entries(run.scenario.recovered_by_truth).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}.
            {run.truth.label === "benign" && run.triage.pred !== "benign" && " This is a false incrimination" + (f.depends_on_recovered ? " caused by recovered records, and it is flagged." : ", not caused by recovered records.")}
          </p>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-rule-2 pt-3 text-sm text-ink-3">
          <ClassTag cls={run.triage.pred} /><span>model {run.model.variant}, rho {run.model.rho}, seed {run.model.seed}</span>
        </div>
      </Card>
    </div>
  );
}
