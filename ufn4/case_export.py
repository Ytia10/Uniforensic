"""Export one case's artifacts to CASE/UCO JSON-LD, and read it back.

Why: N4's claim is that every agency can compute the SAME 27-feature vector from
a community standard (CASE/UCO, Casey et al.), which is what makes horizontal
federated learning valid across agencies. Step 3 proves it two ways:
  (1) the official CASE validator (case_utils.case_validate) accepts our output;
  (2) features computed from the CASE file equal the features computed from the
      original data, for every exported case.

Mapping (unified schema -> CASE/UCO 1.5.0 classes and properties):
  case       -> case-investigation:Investigation (core:name = case id, core:object = artifacts)
  phone/ID   -> observable:ObservableObject + observable:PhoneAccountFacet (phoneNumber)
  sms        -> observable:Message + observable:MessageFacet (from, to, messageText, sentTime,
                messageType "SMS")
  call       -> observable:ObservableObject + observable:CallFacet (from, to, startTime,
                duration in seconds, callType "outgoing"/"incoming"; answered = duration > 0)
  web visit  -> observable:ObservableObject + observable:URLHistoryFacet with one
                observable:URLHistoryEntry (url -> object with observable:URLFacet fullValue)
  login      -> observable:ObservableObject + observable:EventRecordFacet (eventType
                login_success/login_failure, startTime, account, eventRecordServiceName,
                eventRecordText "source_ip_country=..")
  payment    -> observable:ObservableObject + observable:EventRecordFacet (eventType
                payment_in/payment_out, startTime, account = counterparty, eventRecordText "amount_inr=..")
UCO has no dedicated payment or login class, so EventRecordFacet carries both.
The answer key (role, pattern_id) is NOT exported: an agency would not have it.
"""
from __future__ import annotations

import uuid

import pandas as pd

CONTEXT = {
    "kb": "http://example.org/kb/",
    "case-investigation": "https://ontology.caseontology.org/case/investigation/",
    "core": "https://ontology.unifiedcyberontology.org/uco/core/",
    "observable": "https://ontology.unifiedcyberontology.org/uco/observable/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
OWNER = "OWNER-DEVICE"           # the seized phone's own number (not stored in the unified schema)
_NS = uuid.UUID("6f1c3d2e-8a4b-5c6d-9e0f-112233445566")   # fixed namespace -> same case, same IDs


def _iri(kind: str, *key) -> str:
    """CASE asks that every identifier ends with a UUID. uuid5 is deterministic, so
    re-exporting the same case gives byte-identical files."""
    return f"kb:{kind}-{uuid.uuid5(_NS, '|'.join(map(str, key)))}"


def _dt(ts: str) -> dict:
    return {"@type": "xsd:dateTime", "@value": ts}


def _int(v) -> dict:
    return {"@type": "xsd:integer", "@value": int(v)}


def to_case_jsonld(case_id: str, t: dict[str, pd.DataFrame]) -> dict:
    """t = {'sms': df, 'call': df, 'web': df, 'login': df, 'txn': df} for ONE case."""
    graph, objects = [], []
    cache: dict[str, str] = {}

    def node(kind: str, key: str, facet_type: str, facet: dict) -> str:
        """One shared object per phone number / account / URL within the case."""
        k = f"{kind}:{key}"
        if k not in cache:
            oid = _iri(kind, case_id, key)
            graph.append({"@id": oid, "@type": "observable:ObservableObject",
                          "core:hasFacet": [{"@id": _iri("facet", oid), "@type": facet_type, **facet}]})
            cache[k] = oid
        return cache[k]

    phone = lambda n: node("phone", str(n), "observable:PhoneAccountFacet", {"observable:phoneNumber": str(n)})
    account = lambda a: node("account", str(a), "observable:AccountFacet",
                             {"observable:accountIdentifier": str(a)})
    url = lambda d: node("url", str(d), "observable:URLFacet", {"observable:fullValue": f"https://{d}/"})
    me = phone(OWNER)

    def add(aid: str, typ: str, facet_type: str, facet: dict) -> None:
        oid = _iri("artifact", aid)
        graph.append({"@id": oid, "@type": typ,
                      "core:hasFacet": [{"@id": _iri("facet", oid), "@type": facet_type, **facet}]})
        objects.append({"@id": oid})

    for r in t["sms"].itertuples():
        other = phone(r.counterparty)
        src, dst = (me, other) if r.direction == "out" else (other, me)
        add(r.artifact_id, "observable:Message", "observable:MessageFacet",
            {"observable:from": {"@id": src}, "observable:to": [{"@id": dst}],
             "observable:messageText": str(r.text), "observable:sentTime": _dt(r.timestamp),
             "observable:messageType": "SMS"})
    for r in t["call"].itertuples():
        other = phone(r.counterparty)
        src, dst = (me, other) if r.direction == "out" else (other, me)
        add(r.artifact_id, "observable:ObservableObject", "observable:CallFacet",
            {"observable:from": {"@id": src}, "observable:to": [{"@id": dst}],
             "observable:startTime": _dt(r.timestamp), "observable:duration": _int(r.duration_s),
             "observable:callType": "outgoing" if r.direction == "out" else "incoming"})
    for r in t["web"].itertuples():
        u = url(r.domain)
        add(r.artifact_id, "observable:ObservableObject", "observable:URLHistoryFacet",
            {"observable:urlHistoryEntry": [{
                "@id": _iri("url-history-entry", r.artifact_id), "@type": "observable:URLHistoryEntry",
                "observable:url": {"@id": u}, "observable:firstVisit": _dt(r.timestamp),
                "observable:lastVisit": _dt(r.timestamp), "observable:visitCount": _int(1)}]})
    for r in t["login"].itertuples():
        ok = str(r.success).lower() == "true"
        add(r.artifact_id, "observable:ObservableObject", "observable:EventRecordFacet",
            {"observable:eventType": "login_success" if ok else "login_failure",
             "observable:startTime": _dt(r.timestamp), "observable:account": {"@id": account(r.account)},
             "observable:eventRecordServiceName": str(r.service),
             "observable:eventRecordText": f"source_ip_country={r.ip_country}"})
    for r in t["txn"].itertuples():
        add(r.artifact_id, "observable:ObservableObject", "observable:EventRecordFacet",
            {"observable:eventType": "payment_in" if r.direction == "in" else "payment_out",
             "observable:startTime": _dt(r.timestamp),
             "observable:account": {"@id": account(r.counterparty)},
             "observable:eventRecordText": f"amount_inr={float(r.amount)!r}"})

    graph.insert(0, {"@id": _iri("investigation", case_id), "@type": "case-investigation:Investigation",
                     "core:name": case_id, "core:object": objects})
    return {"@context": CONTEXT, "@graph": graph}


def from_case_jsonld(doc: dict) -> dict[str, pd.DataFrame]:
    """Read our CASE JSON-LD back into the unified tables the N1 feature code expects."""
    g = {n["@id"]: n for n in doc["@graph"]}
    inv = next(n for n in doc["@graph"] if n["@type"] == "case-investigation:Investigation")
    cid = inv["core:name"]
    facet = lambda oid: g[oid]["core:hasFacet"][0]
    val = lambda oid, prop: facet(oid)[prop]
    out = {k: [] for k in ("sms", "call", "web", "login", "txn")}
    for ref in inv["core:object"]:
        oid = ref["@id"]
        f = facet(oid)
        ft = f["@type"]
        if ft == "observable:MessageFacet":
            src, dst = f["observable:from"]["@id"], f["observable:to"][0]["@id"]
            out_dir = val(src, "observable:phoneNumber") == OWNER
            other = dst if out_dir else src
            out["sms"].append({"case_id": cid, "timestamp": f["observable:sentTime"]["@value"],
                               "direction": "out" if out_dir else "in",
                               "counterparty": val(other, "observable:phoneNumber"),
                               "text": f["observable:messageText"]})
        elif ft == "observable:CallFacet":
            outgoing = f["observable:callType"] == "outgoing"
            other = f["observable:to"][0]["@id"] if outgoing else f["observable:from"]["@id"]
            dur = int(f["observable:duration"]["@value"])
            out["call"].append({"case_id": cid, "timestamp": f["observable:startTime"]["@value"],
                                "direction": "out" if outgoing else "in",
                                "counterparty": val(other, "observable:phoneNumber"),
                                "answered": dur > 0, "duration_s": dur})
        elif ft == "observable:URLHistoryFacet":
            e = f["observable:urlHistoryEntry"][0]
            full = val(e["observable:url"]["@id"], "observable:fullValue")
            out["web"].append({"case_id": cid, "timestamp": e["observable:firstVisit"]["@value"],
                               "domain": full.removeprefix("https://").rstrip("/")})
        elif ft == "observable:EventRecordFacet":
            et = f["observable:eventType"]
            acct = val(f["observable:account"]["@id"], "observable:accountIdentifier")
            text = f["observable:eventRecordText"].split("=", 1)[1]
            ts = f["observable:startTime"]["@value"]
            if et.startswith("login"):
                out["login"].append({"case_id": cid, "timestamp": ts, "success": et == "login_success",
                                     "account": acct, "service": f["observable:eventRecordServiceName"],
                                     "ip_country": text})
            else:
                out["txn"].append({"case_id": cid, "timestamp": ts,
                                   "direction": "in" if et == "payment_in" else "out",
                                   "counterparty": acct, "amount": float(text)})
    cols = {"sms": ["case_id", "timestamp", "direction", "counterparty", "text"],
            "call": ["case_id", "timestamp", "direction", "counterparty", "answered", "duration_s"],
            "web": ["case_id", "timestamp", "domain"],
            "login": ["case_id", "timestamp", "success", "account", "service", "ip_country"],
            "txn": ["case_id", "timestamp", "direction", "counterparty", "amount"]}
    return {k: pd.DataFrame(v, columns=cols[k]) for k, v in out.items()}
