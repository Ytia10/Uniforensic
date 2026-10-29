import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type RunResult } from "./api";

export type Page =
  | "overview" | "intake" | "recovery" | "triage" | "certainty" | "exchange" | "report"
  | "n1" | "n2" | "n3" | "n4" | "n5";

type AppState = {
  page: Page;
  go: (p: Page) => void;
  reveal: boolean;                 // benchmark mode: show the planted answer key
  setReveal: (v: boolean) => void;
  run: RunResult | null;           // the case analysis the workbench pages show
  setRun: (r: RunResult | null) => void;
};

const Ctx = createContext<AppState | null>(null);

const readHash = (): Page => (window.location.hash.replace("#/", "") || "overview") as Page;

export function AppProvider({ children }: { children: ReactNode }) {
  const [page, setPage] = useState<Page>(readHash());
  const [reveal, setRevealState] = useState(() => localStorageGet("uf.reveal") === "1");
  const [run, setRunState] = useState<RunResult | null>(null);

  useEffect(() => {
    const on = () => setPage(readHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  // reopen the last analysed case after a page reload
  useEffect(() => {
    const id = localStorageGet("uf.run");
    if (id) api.run(id).then(setRunState).catch(() => localStorageSet("uf.run", ""));
  }, []);

  const go = (p: Page) => { window.location.hash = `#/${p}`; document.getElementById("main")?.scrollTo(0, 0); };
  const setReveal = (v: boolean) => { setRevealState(v); localStorageSet("uf.reveal", v ? "1" : "0"); };
  const setRun = (r: RunResult | null) => { setRunState(r); localStorageSet("uf.run", r?.run_id ?? ""); };
  return <Ctx.Provider value={{ page, go, reveal, setReveal, run, setRun }}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}

// localStorage can be unavailable (private windows); the app works without it.
function localStorageGet(k: string): string | null {
  try { return window.localStorage.getItem(k); } catch { return null; }
}
function localStorageSet(k: string, v: string) {
  try { window.localStorage.setItem(k, v); } catch { /* ignore */ }
}
