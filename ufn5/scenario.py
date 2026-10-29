"""Deletion scenarios: which artifacts a case's user deleted, when, and what foreign
residue the device holds. Every random choice is seeded per case, so the same case
gets the SAME background deletions and the SAME evidence ordering at every level
(levels are paired: 25% deleted is a subset of 50% deleted).

level s (share of the case's planted decisive evidence the suspect deleted):
    the first round(s * n) artifacts of a fixed random permutation of the evidence.
background deletions (all cases, all levels): each background or distractor
    artifact is deleted with probability `background_delete_prob` - ordinary people
    delete ordinary messages too, so 'something was recovered' does not mean 'guilty'.
timing: background records are deleted 0-30 days after they were created. The
    suspect (and, separately, the foreign owner) follows ONE of two behaviours per
    case, drawn at random:
      'immediate' - deletes each incriminating record 0-72 h after creating it
      'clean-up'  - deletes them all in the last 48 h before the device is seized
    Deletions not done by acquisition time are done just before it. The device keeps
    being used, so later inserts can overwrite deleted records (SQLite reuses free
    space on the page that is still being filled).
foreign residue (benign cases, 'injected' version only): records that are really
    on the device and really deleted, but do NOT belong to the suspect (shared or
    second-hand phone). The carver reconstructs them like any other deleted record:
    a false recovery that no carving confidence can catch. Generated with N1's own
    pattern code (ufn1.generator.CaseBuilder.plant), role = 'foreign'.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ufn1.common import derive_seed
from ufn1.generator import CaseBuilder
from ufn1.schema import CLASSES, PATTERNS

from .devicedb import TABLES, to_epoch, to_row


def case_start(tables: dict[str, pd.DataFrame]) -> datetime:
    ts = min(d.timestamp.min() for d in tables.values() if len(d))
    return datetime.fromisoformat(ts).replace(hour=0, minute=0, second=0)


def foreign_residue(case_id: str, tables: dict[str, pd.DataFrame], scfg: dict, seed: int,
                    window_days: int) -> dict[str, pd.DataFrame]:
    """Planted-pattern artifacts of ANOTHER person, in the same time window."""
    rng = np.random.default_rng(derive_seed(seed, "foreign", case_id))
    cb = CaseBuilder(f"{case_id}-FOREIGN", rng, window_days)
    cb.start = case_start(tables)
    cls = str(rng.choice([c for c in CLASSES if c != "benign"]))
    lo, hi = scfg["foreign_intensity"]
    for p in rng.choice(list(PATTERNS[cls]), size=scfg["foreign_patterns"], replace=False):
        cb.plant(str(p), int(rng.integers(lo, hi + 1)), "foreign")
    out = {}
    for t in TABLES:
        df = pd.DataFrame(cb.rows[t])
        if len(df):
            df["case_id"] = case_id
        out[t] = df
    return out


def plan(case_id: str, tables: dict[str, pd.DataFrame], level: float, foreign: bool,
         scfg: dict, seed: int, window_days: int):
    """Return (events_by_table, truth) for one case version.
    events_by_table[t] = time-ordered [('ins', artifact_id, row) | ('del', artifact_id, None)]
    truth = DataFrame: artifact_id, table, role, pattern_id, owner ('suspect'|'foreign'),
            deleted (bool), row (stored tuple)."""
    rng = np.random.default_rng(derive_seed(seed, "scenario", case_id))   # same for every level
    recs = []
    for t in TABLES:
        for a in tables[t].itertuples(index=False):
            recs.append((t, a, "suspect"))
    if foreign:
        for t, df in foreign_residue(case_id, tables, scfg, seed, window_days).items():
            for a in df.itertuples(index=False):
                recs.append((t, a, "foreign"))

    n = len(recs)
    u_bg = rng.random(n)                              # background deletion draws
    delay_bg = rng.uniform(0, scfg["background_delay_days"] * 86400, n)
    delay_ev = rng.uniform(0, scfg["evidence_delay_hours"] * 3600, n)
    ev_idx = [i for i, (_, a, o) in enumerate(recs) if o == "suspect" and a.role == "evidence"]
    perm = rng.permutation(len(ev_idx))
    n_del_ev = int(np.floor(level * len(ev_idx) + 0.5))
    del_ev = {ev_idx[j] for j in perm[:n_del_ev]}
    immediate = {"suspect": rng.random() < scfg["immediate_share"],
                 "foreign": rng.random() < scfg["immediate_share"]}
    before_acq = rng.uniform(0, scfg["cleanup_hours"] * 3600, n)

    acq = max(to_epoch(a.timestamp) for _, a, _ in recs) + 3600   # acquisition time
    events = {t: [] for t in TABLES}
    rows_truth = []
    for i, (t, a, owner) in enumerate(recs):
        ts = to_epoch(a.timestamp)
        row = to_row(t, a)
        incriminating = owner == "foreign" or a.role == "evidence"
        if incriminating:
            deleted = owner == "foreign" or i in del_ev
            t_del = ts + delay_ev[i] if immediate[owner] else acq - before_acq[i]
        else:
            deleted = bool(u_bg[i] < scfg["background_delete_prob"])
            t_del = ts + delay_bg[i]
        events[t].append((ts, 0, i, ("ins", a.artifact_id, row)))
        if deleted:
            t_del = min(max(t_del, ts + 1), acq - 1)
            events[t].append((t_del, 1, i, ("del", a.artifact_id, None)))
        rows_truth.append({"artifact_id": a.artifact_id, "table": t, "role": a.role,
                           "pattern_id": a.pattern_id, "owner": owner, "deleted": deleted,
                           "behaviour": "immediate" if immediate[owner] else "clean-up",
                           "row": row})
    for t in TABLES:                                  # time order; insert before delete on ties
        events[t] = [e[3] for e in sorted(events[t], key=lambda e: (e[0], e[1], e[2]))]
    return events, pd.DataFrame(rows_truth)
