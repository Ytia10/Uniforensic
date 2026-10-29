import { useEffect, useState } from "react";
import { api } from "../api";

/** Load /api/results/<n> once. `data.ok === false` means that novelty's outputs are missing. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function useResults(n: string): { data: any | null; error: string | null } {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.results(n).then((d) => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const r = d as any;
      if (r.ok === false) setError(r.missing);
      setData(r);
    }).catch((e) => setError(e.message));
  }, [n]);
  return { data, error };
}
