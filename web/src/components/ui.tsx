import { useState, type ReactNode } from "react";
import type { Job } from "../api";
import { CLASS_COLOR } from "../theme";

/* ---------------------------------------------------------------- page structure */

/** Page title block. `step` = position in the six-step case procedure; `uses` = novelties involved. */
export function PageHeader({ step, title, lead, uses, right }:
  { step?: number; title: string; lead?: ReactNode; uses?: string; right?: ReactNode }) {
  return (
    <header className="mb-7 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
      <div className="min-w-0 max-w-[46rem]">
        <h1 className="text-[2rem] font-semibold leading-[1.1] text-ink">
          {step != null && <span className="mr-3 font-medium text-ink-4 num">{step}</span>}{title}
        </h1>
        {lead && <p className="mt-3 text-base leading-relaxed text-ink-2">{lead}</p>}
        {uses && <p className="mt-2 text-sm text-ink-3">{uses}</p>}
      </div>
      {right && <div className="no-print shrink-0">{right}</div>}
    </header>
  );
}

/** A working surface: a table, a chart, a form. */
export function Card({ title, subtitle, children, right, className = "", tone = "plain" }:
  { title?: ReactNode; subtitle?: ReactNode; children: ReactNode; right?: ReactNode; className?: string;
    tone?: "plain" | "well" }) {
  return (
    <section className={`print-area rounded-md border border-rule ${tone === "well" ? "bg-well" : "bg-sheet"} px-6 py-5 ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
          <div className="min-w-0 max-w-[44rem]">
            {title && <h2 className="text-[1.2rem] font-semibold leading-snug text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-sm leading-normal text-ink-2">{subtitle}</p>}
          </div>
          {right && <div className="no-print">{right}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/** A row of key figures. Children are <Stat>s; they sit side by side, separated by rules. */
export function StatRow({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-y-5 rounded-md border border-rule bg-sheet py-4 sm:grid-cols-3 lg:auto-cols-fr lg:grid-flow-col lg:grid-cols-none [&>*]:border-rule-2 [&>*]:px-5 lg:[&>*+*]:border-l">
      {children}
    </div>
  );
}

const TONE_TEXT = { default: "text-ink", good: "text-good", warn: "text-warn", bad: "text-bad", rec: "text-rec" } as const;

/** One key figure: the number first, then what it counts. */
export function Stat({ label, value, note, tone = "default" }:
  { label: ReactNode; value: ReactNode; note?: ReactNode; tone?: keyof typeof TONE_TEXT | "accent" }) {
  const t = tone === "accent" ? "rec" : tone;
  return (
    <div className="min-w-0">
      <div className={`font-display text-[1.9rem] font-semibold leading-none num ${TONE_TEXT[t]}`}>{value}</div>
      <div className="mt-2 text-sm font-semibold leading-snug text-ink">{label}</div>
      {note && <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{note}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- small marks */

const BADGE = {
  neutral: "border-rule text-ink-2 bg-sheet",
  good: "border-good/40 text-good bg-good-tint",
  warn: "border-warn/40 text-warn bg-warn-tint",
  bad: "border-bad/40 text-bad bg-bad-tint",
  accent: "border-rec/40 text-rec bg-rec-tint",
} as const;

/** A status word. Use only for a real state (validated, flagged, identical...). */
export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: keyof typeof BADGE }) {
  return <span className={`inline-flex items-center whitespace-nowrap rounded-xs border px-1.5 py-px text-[13px] font-medium leading-5 ${BADGE[tone]}`}>{children}</span>;
}

/** A case class (benign, fraud, ...): colour square + name. Not a status, so no fill. */
export function ClassTag({ cls, prefix }: { cls: string; prefix?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm font-medium text-ink">
      <span className="inline-block h-2.5 w-2.5 rounded-[1px]" style={{ background: CLASS_COLOR[cls] ?? "#6f7c77" }} />
      {prefix}{cls}
    </span>
  );
}

/** Legend keys for the three kinds of data. */
export function DataKey({ kind, children }: { kind: "live" | "deleted" | "recovered"; children: ReactNode }) {
  const sw = kind === "live" ? "solid-live" : kind === "deleted" ? "outline-deleted" : "hatch-rec";
  return (
    <span className="inline-flex items-center gap-2 text-sm text-ink-2">
      <span className={`inline-block h-3 w-5 rounded-[1px] ${sw}`} />{children}
    </span>
  );
}

// kept for pages that colour a class name; maps to badge tones
export const CLASS_TONE: Record<string, "good" | "bad" | "warn" | "accent"> = {
  benign: "good", fraud: "bad", harassment: "warn", cybercrime: "accent",
};

/* ---------------------------------------------------------------- controls */

export function Button({ children, onClick, disabled, kind = "primary", type = "button" }:
  { children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: "primary" | "secondary" | "quiet"; type?: "button" | "submit" }) {
  const cls = {
    primary: "bg-ink text-white hover:bg-live disabled:bg-ink-4",
    secondary: "border border-rule bg-sheet text-ink hover:border-ink-3 disabled:text-ink-4",
    quiet: "px-0 text-ink underline decoration-ink-4 underline-offset-2 hover:decoration-ink disabled:text-ink-4",
  }[kind];
  return (
    <button type={type} onClick={onClick} disabled={disabled}
            className={`inline-flex items-center gap-2 whitespace-nowrap rounded-sm ${kind === "quiet" ? "" : "px-3.5 py-[7px]"} text-[15px] font-semibold disabled:cursor-not-allowed ${cls}`}>
      {children}
    </button>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-semibold text-ink">{label}</span>
      {children}
    </label>
  );
}

export const selectCls = "w-full rounded-sm border border-rule bg-sheet px-2.5 py-[7px] text-[15px] text-ink hover:border-ink-3 focus:border-ink focus:outline-none";
export const checkCls = "h-4 w-4 accent-[#17201d]";

/* ---------------------------------------------------------------- states */

export function ErrorBox({ error }: { error: string | null | undefined }) {
  if (!error) return null;
  return <div role="alert" className="rounded-sm border border-bad/40 bg-bad-tint px-4 py-3 text-sm text-bad">{error}</div>;
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="flex items-center gap-3 py-10 text-sm text-ink-3"><Spinner />{what}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed border-ink-4 bg-sheet px-8 py-10 text-[15px] text-ink-2">{children}</div>;
}

function Spinner() {
  return <span className="spin inline-block h-3.5 w-3.5 rounded-full border-2 border-rule border-t-ink" aria-hidden />;
}

function StepMark({ state }: { state: "running" | "done" | "error" }) {
  if (state === "running") return <span className="mt-[3px]"><Spinner /></span>;
  if (state === "error") return <span className="mt-px w-4 text-center font-semibold text-bad" aria-label="failed">×</span>;
  return (
    <svg className="mt-[3px] h-4 w-4 shrink-0 text-good" viewBox="0 0 16 16" aria-label="done">
      <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

/** Live view of a background job: its steps, timings and log. */
export function JobPanel({ job, showLog = false }: { job: Job | null; showLog?: boolean }) {
  const [open, setOpen] = useState(showLog);
  if (!job) return null;
  const word = { queued: "Waiting for the previous job", running: "Running", done: "Finished", error: "Stopped with an error" }[job.status];
  return (
    <div className="rounded-sm border border-rule-2 bg-well px-4 py-3.5">
      <div className="mb-3 flex items-center justify-between gap-4 text-sm">
        <span className={`font-semibold ${job.status === "error" ? "text-bad" : job.status === "done" ? "text-good" : "text-ink"}`}>
          {word} <span className="font-normal text-ink-3 num">in {job.seconds.toFixed(1)} s</span>
        </span>
        {job.lines.length > 0 && (
          <button className="text-sm text-ink-2 underline decoration-ink-4 underline-offset-2 hover:text-ink" onClick={() => setOpen(!open)}>
            {open ? "Hide log" : "Show log"}
          </button>
        )}
      </div>
      {job.progress > 0 && job.progress < 1 && (
        <div className="mb-3 h-1 w-full bg-rule-2"><div className="h-full bg-ink" style={{ width: `${job.progress * 100}%` }} /></div>
      )}
      <ol className="space-y-2.5">
        {job.steps.map((s, i) => (
          <li key={i} className="flex gap-3 text-[15px]">
            <StepMark state={s.state} />
            <div className="min-w-0 flex-1">
              <div className="text-ink">{s.name}</div>
              {s.detail && <div className="text-sm text-ink-3">{s.detail}</div>}
            </div>
            {s.seconds != null && <span className="shrink-0 text-sm text-ink-3 num">{s.seconds} s</span>}
          </li>
        ))}
      </ol>
      {job.error && <div className="mt-3"><ErrorBox error={job.error} /></div>}
      {open && job.lines.length > 0 && (
        <pre className="mt-3 max-h-64 overflow-auto rounded-xs bg-ink p-3 font-mono text-xs leading-5 text-white">{job.lines.join("\n")}</pre>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- figures and tables */

export function FigureImg({ src, caption }: { src: string; caption: string }) {
  return (
    <figure className="rounded-md border border-rule bg-sheet p-3">
      <a href={src} target="_blank" rel="noreferrer" className="block"><img src={src} alt={caption} className="mx-auto w-full" loading="lazy" /></a>
      <figcaption className="mt-2 font-mono text-xs text-ink-3">{caption}</figcaption>
    </figure>
  );
}

export function Th({ children, right }: { children?: ReactNode; right?: boolean }) {
  return <th scope="col" className={`border-b border-ink/70 px-3 pb-2 pt-1 text-sm font-semibold text-ink ${right ? "text-right" : "text-left"}`}>{children}</th>;
}
export function Td({ children, right, mono, className = "" }: { children?: ReactNode; right?: boolean; mono?: boolean; className?: string }) {
  return <td className={`border-b border-rule-2 px-3 py-2 align-top text-[15px] ${right ? "text-right num" : ""} ${mono ? "font-mono text-[13px]" : ""} ${className}`}>{children}</td>;
}
