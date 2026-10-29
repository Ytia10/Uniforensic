// Typed client for the FastAPI backend (ufapp/main.py). Every number on screen comes from here.

export type Job = {
  id: string;
  kind: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error: string | null;
  seconds: number;
  steps: { name: string; state: "running" | "done" | "error"; seconds: number | null; detail: string }[];
  lines: string[];
  has_result: boolean;
};

export type Headline = { novelty: string; title: string; value: number; fmt: "pct" | "num3"; note: string };
export type IntegrityItem = { status: string; checked: number; ok: number; changed: string[]; missing: string[]; note?: string };
export type Overview = {
  version: string;
  available: Record<string, boolean>;
  missing: Record<string, string>;
  integrity: { checked_at: string; novelties: Record<string, IntegrityItem> };
  headlines: Headline[];
};

export type ModelInfo = { variant: string; rho: number; seed: number; n_inputs: number };
export type CaseRow = {
  case_id: string; label: string; tier: string; n_artifacts: number;
  by_table: Record<string, number>; stored_versions: string[];
};
export type Artifact = { table: string; timestamp: string; artifact_id: string; what: string; role: string; pattern: string };
export type CaseDetail = {
  case_id: string; label: string; tier: string; planted_features: string[];
  evidence_patterns: string[]; distractor_patterns: string[]; artifacts: Artifact[];
};

export type RecoveredRecord = {
  table: string; table_label: string; timestamp: string; what: string; p_raw: number; confidence: number;
  strategy: string; source_type: string; teammate_fuzzy_filter_drops: boolean; truth: string; truth_role: string | null;
};
export type RunResult = {
  run_id: string; case_id: string; version_id: string; label_truth: string; tier_truth: string;
  scenario: {
    level: number; foreign: boolean; behaviour_requested: string; behaviour: string; n_artifacts: number;
    n_deleted: number; n_evidence: number; n_evidence_deleted: number; n_foreign: number; n_recovered: number;
    recovered_by_truth: Record<string, number>; databases: string[];
  };
  model: { variant: string; rho: number; seed: number; n_inputs: number };
  triage: { classes: string[]; proba: number[]; proba_without_recovered: number[]; pred: string;
            pred_without_recovered: string; confidence: number };
  flag: { depends_on_recovered: boolean; recovered_attribution_share: number | null };
  sets: { alpha: number; q_full: number; q_live: number; with_recovered: string[]; live_only: string[] };
  explanation: {
    method: string;
    top_inputs: { name: string; label: string; attribution: number; channel: "live" | "recovered"; value: number }[];
    top_base_features: { name: string; label: string; attribution_abs: number; planted_and_visible: boolean }[];
    acceptance: { validated: boolean; reasons: string[]; agreement_top5: number | null; tau: number;
                  min_agreement: number; kernelshap_top5: string[] | null };
  };
  decisive_features: { name: string; label: string; live: number; merged_A: number; weighted_B: number;
                       recovered_channel: number; planted: boolean; visible: boolean }[];
  recovered_records: RecoveredRecord[];
  timeline: { day: string; live: number; deleted: number; recovered: number }[];
  case_export: { file: string; sha256: string; objects: number };
  consistency: { in_n5_grid: boolean; stored_version: string | null; identical?: boolean; summary: string;
                 max_feature_diff?: Record<string, number>; max_proba_diff?: number };
  truth: { label: string; tier: string; planted_features: string[]; visible_features: string[]; evidence_patterns: string[] };
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) =>
  call<T>(path, { method: "POST", headers: { "Content-Type": "application/json" },
                  body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  overview: (refresh = false) => call<Overview>(`/api/overview${refresh ? "?refresh=true" : ""}`),
  models: () => call<{ models: ModelInfo[]; default: { variant: string; rho: number; seed: number }; levels: number[] }>("/api/models"),
  cases: () => call<{ cases: CaseRow[] }>("/api/cases"),
  caseDetail: (id: string) => call<CaseDetail>(`/api/cases/${id}`),
  analyse: (body: Record<string, unknown>) => post<Job>("/api/analyse", body),
  job: (id: string) => call<Job>(`/api/jobs/${id}`),
  jobResult: <T,>(id: string) => call<T>(`/api/jobs/${id}/result`),
  runs: () => call<{ runs: { run_id: string; case_id: string; version_id: string; pred: string; flag: boolean;
                            model: ModelInfo; when: string }[] }>("/api/runs"),
  run: (id: string) => call<RunResult>(`/api/runs/${id}`),
  validate: (runId: string) => post<Job>(`/api/runs/${runId}/validate`),
  results: <T,>(n: string) => call<T>(`/api/results/${n}`),
  n3Pipeline: () => post<Job>("/api/n3/pipeline"),
  n3Upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<Job>("/api/n3/upload", { method: "POST", body: fd });
  },
  fedRun: (body: Record<string, unknown>) => post<Job>("/api/fed/run", body),
  privacyPrepare: () => post<Job>("/api/privacy/prepare"),
  privacyRelease: (eps: string, repeat: number) =>
    call<{ eps: number | null; repeat: number; rows: { class: string; exact_reliance: number; released_reliance: number;
                                                         abs_error: number; evidence_recall_at_3: number }[] }>(
      `/api/privacy/release?eps=${eps}&repeat=${repeat}`),
};

export const figure = (n: string, name: string) => `/api/figures/${n}/${name}`;
export const runFile = (runId: string, rel: string) => `/api/runs/${runId}/files/${rel}`;

/** Poll a job until it finishes; calls onUpdate on every poll. */
export async function follow(job: Job, onUpdate: (j: Job) => void, everyMs = 400): Promise<Job> {
  let j = job;
  onUpdate(j);
  while (j.status === "queued" || j.status === "running") {
    await new Promise((r) => setTimeout(r, everyMs));
    j = await api.job(j.id);
    onUpdate(j);
  }
  return j;
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "-" : `${(x * 100).toFixed(d)}%`);
export const num = (x: number | null | undefined, d = 3) => (x == null ? "-" : x.toFixed(d));

/** The consistency sentence, plus why the stored version id can differ from this run's (benign cases). */
export const consistencyText = (run: RunResult, reveal: boolean) => {
  const c = run.consistency;
  if (!c.stored_version || c.stored_version === run.version_id) return c.summary;
  return c.summary + (reveal
    ? ". A benign case has no planted evidence, so every deletion share gives the same phone; N5 stored it once, at share 0."
    : ". N5 stored this phone under an equivalent scenario.");
};
