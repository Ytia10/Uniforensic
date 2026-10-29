import { useEffect, useState, type ReactNode } from "react";
import { useApp, type Page } from "./state";
import { Badge, ClassTag, checkCls } from "./components/ui";
import { C } from "./theme";
import Overview from "./pages/Overview";
import Intake from "./pages/Intake";
import Recovery from "./pages/Recovery";
import Triage from "./pages/Triage";
import Certainty from "./pages/Certainty";
import Exchange from "./pages/Exchange";
import Report from "./pages/Report";
import N1Page from "./pages/N1Page";
import N2Page from "./pages/N2Page";
import N3Page from "./pages/N3Page";
import N4Page from "./pages/N4Page";
import N5Page from "./pages/N5Page";

const STEPS: { page: Page; label: string }[] = [
  { page: "intake", label: "Case intake" },
  { page: "recovery", label: "Deleted-data recovery" },
  { page: "triage", label: "Triage and explanation" },
  { page: "certainty", label: "Provenance and certainty" },
  { page: "exchange", label: "Evidence exchange" },
  { page: "report", label: "Case report" },
];
const NOVELTIES: { page: Page; code: string; label: string }[] = [
  { page: "n1", code: "N1", label: "Validated explanations" },
  { page: "n2", code: "N2", label: "Private explanation sharing" },
  { page: "n3", code: "N3", label: "SQLite carving" },
  { page: "n4", code: "N4", label: "Federated triage" },
  { page: "n5", code: "N5", label: "Recovered-evidence provenance" },
];

const PAGES: Record<Page, () => ReactNode> = {
  overview: () => <Overview />, intake: () => <Intake />, recovery: () => <Recovery />,
  triage: () => <Triage />, certainty: () => <Certainty />, exchange: () => <Exchange />,
  report: () => <Report />, n1: () => <N1Page />, n2: () => <N2Page />, n3: () => <N3Page />,
  n4: () => <N4Page />, n5: () => <N5Page />,
};

// Pages with live runs stay mounted once opened, so a finished run is still there when you come back.
const KEEP: Page[] = ["n2", "n3", "n4"];

/** SVG patterns referenced as url(#...) by every chart. Must not be display:none. */
function Patterns() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden>
      <defs>
        <pattern id="hatch-rec" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" fill={C.recTint} />
          <rect width="2.2" height="6" fill={C.rec} />
        </pattern>
      </defs>
    </svg>
  );
}

function NavItem({ active, onClick, lead, children }: { active: boolean; onClick: () => void; lead: ReactNode; children: ReactNode }) {
  return (
    <button onClick={onClick} aria-current={active ? "page" : undefined}
            className={`flex w-full items-baseline gap-3 rounded-sm px-2.5 py-[7px] text-left text-[15px] ${active ? "bg-sheet font-semibold text-ink shadow-[inset_0_0_0_1px_var(--color-rule)]" : "text-ink-2 hover:bg-sheet/60 hover:text-ink"}`}>
      <span className={`w-6 shrink-0 font-display text-[15px] num ${active ? "text-ink" : "text-ink-3"}`}>{lead}</span>
      <span>{children}</span>
    </button>
  );
}

export default function App() {
  const { page, go, reveal, setReveal, run } = useApp();
  const [seen, setSeen] = useState<Page[]>([]);
  useEffect(() => { if (KEEP.includes(page)) setSeen((s) => (s.includes(page) ? s : [...s, page])); }, [page]);
  return (
    <div className="flex h-screen overflow-hidden print:block print:h-auto print:overflow-visible">
      <Patterns />
      <aside className="no-print hidden h-screen w-[17.5rem] shrink-0 flex-col border-r border-rule lg:flex">
        <button onClick={() => go("overview")} className="block px-5 pb-5 pt-6 text-left">
          <div className="font-display text-[1.35rem] font-semibold leading-none text-ink">UniForensic Workbench</div>
          <div className={`mt-1.5 text-sm ${page === "overview" ? "font-semibold text-ink" : "text-ink-3"}`}>Overview and integrity check</div>
        </button>
        <nav className="flex-1 overflow-y-auto px-3 pb-4" aria-label="Main">
          <h2 className="px-2.5 pb-1.5 pt-2 font-sans text-sm font-semibold text-ink">Analyse a phone</h2>
          {STEPS.map((s, i) => (
            <NavItem key={s.page} active={page === s.page} onClick={() => go(s.page)} lead={i + 1}>{s.label}</NavItem>
          ))}
          <h2 className="px-2.5 pb-1.5 pt-6 font-sans text-sm font-semibold text-ink">Evidence for each novelty</h2>
          {NOVELTIES.map((n) => (
            <NavItem key={n.page} active={page === n.page} onClick={() => go(n.page)} lead={n.code}>{n.label}</NavItem>
          ))}
        </nav>
        <div className="space-y-4 border-t border-rule px-5 py-4">
          {run && (
            <button onClick={() => go("triage")} className="block w-full text-left">
              <div className="text-sm text-ink-3">Case on the bench</div>
              <div className="mt-0.5 font-mono text-[13px] text-ink">{run.version_id}</div>
              <div className="mt-1.5 flex flex-wrap items-center gap-2">
                <ClassTag cls={run.triage.pred} />
                {run.flag.depends_on_recovered && <Badge tone="accent">rests on recovered</Badge>}
              </div>
            </button>
          )}
          <label className="flex cursor-pointer items-start gap-3">
            <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} className={`${checkCls} mt-0.5`} />
            <span>
              <span className="block text-[15px] font-semibold text-ink">Show ground truth</span>
              <span className="block text-sm leading-snug text-ink-3">The benchmark's answer key. Leave off for the examiner's view.</span>
            </span>
          </label>
        </div>
      </aside>

      <main id="main" className="min-w-0 flex-1 overflow-y-auto print:overflow-visible">
        <div className="no-print flex items-center gap-3 border-b border-rule bg-sheet px-4 py-2.5 lg:hidden">
          <span className="font-display text-lg font-semibold">UniForensic</span>
          <select aria-label="Go to page" value={page} onChange={(e) => go(e.target.value as Page)}
                  className="min-w-0 flex-1 rounded-sm border border-rule bg-sheet px-2 py-1.5 text-[15px]">
            <option value="overview">Overview</option>
            {STEPS.map((s, i) => <option key={s.page} value={s.page}>{i + 1}. {s.label}</option>)}
            {NOVELTIES.map((n) => <option key={n.page} value={n.page}>{n.code} {n.label}</option>)}
          </select>
          <label className="flex items-center gap-1.5 text-sm"><input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} className={checkCls} />Truth</label>
        </div>
        <div className="mx-auto max-w-[88rem] px-4 py-8 sm:px-8 lg:px-10 print:max-w-none print:p-0">
          {!KEEP.includes(page) && (PAGES[page]?.() ?? <Overview />)}
          {KEEP.filter((p) => p === page || seen.includes(p)).map((p) => <div key={p} hidden={p !== page}>{PAGES[p]()}</div>)}
        </div>
      </main>
    </div>
  );
}
