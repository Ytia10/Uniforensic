"""CASE/UCO export WITH provenance: N4's exporter, plus two UCO 1.5.0 facets on every
recovered artifact (checked against the ontology shipped in case-utils 0.18.0):

  observable:RecoveredObjectFacet   observable:contentRecoveredStatus = "recovered"
                                    (value from UCO's RecoveredObjectStatusVocab:
                                     recovered | partially recovered | overwritten | unknown)
  core:ConfidenceFacet              core:confidence = calibrated confidence in whole
                                    percent (UCO: xsd:nonNegativeInteger, at most 100)

Live artifacts carry neither facet. N4's ufn4.case_export is reused unchanged: the
facets are appended after N4 builds the graph, and N4's reader still works because
it reads the first facet of each object, which is always N4's own.
"""
from __future__ import annotations

import pandas as pd

from ufn4.case_export import _iri, from_case_jsonld, to_case_jsonld

TYPE_OF_FACET = {"observable:MessageFacet": "sms", "observable:CallFacet": "call",
                 "observable:URLHistoryFacet": "web"}


def export_with_provenance(case_id: str, live: dict[str, pd.DataFrame],
                           recovered: dict[str, pd.DataFrame]) -> dict:
    """live/recovered: per-table frames with N1 columns; recovered also has artifact_id
    and confidence_pct (int 0-100)."""
    tabs = {t: pd.concat([live[t], recovered[t]], ignore_index=True) for t in live}
    doc = to_case_jsonld(case_id, tabs)
    nodes = {n["@id"]: n for n in doc["@graph"]}
    for t, r in recovered.items():
        for aid, pct in zip(r.artifact_id, r.confidence_pct):
            oid = _iri("artifact", aid)
            nodes[oid]["core:hasFacet"] += [
                {"@id": _iri("facet-recovered", oid), "@type": "observable:RecoveredObjectFacet",
                 "observable:contentRecoveredStatus": "recovered"},
                {"@id": _iri("facet-confidence", oid), "@type": "core:ConfidenceFacet",
                 "core:confidence": {"@type": "xsd:nonNegativeInteger", "@value": int(pct)}}]
    return doc


def read_with_provenance(doc: dict) -> dict[str, pd.DataFrame]:
    """N4's reader + a weight column: 1 for live artifacts, confidence/100 for recovered."""
    tabs = from_case_jsonld(doc)
    g = {n["@id"]: n for n in doc["@graph"]}
    inv = next(n for n in doc["@graph"] if n["@type"] == "case-investigation:Investigation")
    weights = {t: [] for t in tabs}
    rec = {t: [] for t in tabs}
    for ref in inv["core:object"]:
        facets = g[ref["@id"]]["core:hasFacet"]
        main = facets[0]["@type"]
        if main == "observable:EventRecordFacet":
            t = "login" if facets[0]["observable:eventType"].startswith("login") else "txn"
        else:
            t = TYPE_OF_FACET[main]
        types = [f["@type"] for f in facets]
        conf = [f for f in facets if f["@type"] == "core:ConfidenceFacet"]
        weights[t].append(conf[0]["core:confidence"]["@value"] / 100 if conf else 1.0)
        rec[t].append("observable:RecoveredObjectFacet" in types)
    return {t: df.assign(w=weights[t], recovered=rec[t]) for t, df in tabs.items()}
