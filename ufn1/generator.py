"""Synthetic case generator with PLANTED decisive evidence.

Idea (EviPlant, Scanlon et al. 2017; TraceGen, Du et al. 2021): build realistic
background activity, then inject known evidence. The injection log is the answer
key. Here we plant at the artifact level (SMS, calls, web, logins, transactions);
the feature step later turns artifacts into one case-level vector.

All domains use the reserved '.example' TLD and all names are fictional.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .schema import CLASSES, PATTERNS, PATTERN_TO_FEATURE

# ------------------------------------------------------------------ text templates
PERSONAL = ["Reached home", "Call me when free", "Let's meet at 6", "Happy birthday!",
            "Did you eat?", "Send me the notes", "Running late, 10 mins", "Ok see you",
            "Where are you?", "Thanks a lot", "Good night", "Pick up milk please"]
PERSONAL_NOISE = ["Got the refund for the shoes?",            # looks like scam keyword 'refund'
                  "My insta account got blocked lol",           # 'blocked'
                  "You will regret not coming, lol"]            # looks abusive
WORK = ["Meeting moved to 3pm", "Please share the report", "Invoice sent by mail",
        "Can we push the call?", "Approved, go ahead"]
PROMO = ["Flat 50% off today only", "You will regret missing this sale!",
         "Your order has shipped", "Win a prize on every purchase", "New offers inside"]
BANK = ["Your OTP is {n6}. Do not share it with anyone.",
        "Rs {amt} debited from a/c XX{n4}", "Rs {amt} credited to a/c XX{n4}"]
SCAM = ["Your bank account will be blocked today. Update KYC at {link}",
        "Share OTP to receive your refund of Rs {amt}",
        "Congratulations! You won a prize of Rs {amt}. Verify your account at {link}",
        "Your SIM will be deactivated. Complete KYC now: {link}"]
BLAST = ["Hi, are you free to talk?", "Hello, saw your profile, please reply",
         "Hi dear, is this the right number?", "Hello, are you interested in a job?"]
ABUSIVE = ["I know where you live", "You will regret ignoring me",
           "Answer my calls or else", "Watch your back", "You are worthless"]

BENIGN_DOMAINS = ["news-daily.example", "shopmart.example", "socialbook.example",
                  "vidstream.example", "edu-portal.example", "mailhub.example",
                  "chatgram.example", "travelhub.example", "recipes.example"]
TECH_DOMAINS = ["devforum.example", "codehub.example"]
HACK_DOMAINS = ["exploit-archive.example", "booter-stress.example", "credleaks.example",
                "rat-builder.example", "carding-forum.example"]
COUNTRIES = ["IN", "US", "NL", "DE", "SG", "RU", "BR", "FR", "GB", "JP",
             "UA", "RO", "TR", "VN", "CA"]
SERVICES = ["mail", "social", "bank", "work"]

# hour-of-day weights for normal activity (little at night)
_H = np.array([1, 1, 0.5, 0.5, 0.5, 1, 3, 5, 6, 6, 6, 6,
               6, 6, 6, 6, 6, 6, 7, 7, 7, 6, 5, 3], dtype=float)
HOUR_P = _H / _H.sum()


class CaseBuilder:
    """Builds all artifacts of one case. One instance = one case."""

    def __init__(self, case_id: str, rng: np.random.Generator, window_days: int):
        self.case_id = case_id
        self.rng = rng
        self.window_days = window_days
        self.start = datetime(2026, 1, 1) + timedelta(days=int(rng.integers(0, 300)))
        self.rows = {t: [] for t in ["sms", "call", "web", "login", "txn"]}
        self.contacts = [self._phone() for _ in range(int(rng.integers(15, 51)))]
        w = 1.0 / np.arange(1, len(self.contacts) + 1)       # Zipf: few close contacts
        self.contact_p = w / w.sum()
        self.payees = [f"payee-{rng.integers(1000, 9999)}" for _ in range(int(rng.integers(4, 10)))]
        self.user = f"user{rng.integers(10000, 99999)}"
        self.is_tech = rng.random() < 0.2                     # tech-savvy persona -> noise
        self.travels = rng.random() < 0.1                     # travel persona -> noise

    # ---------------------------------------------------------- small helpers
    def _phone(self) -> str:
        r = self.rng
        return "+91" + str(int(r.integers(6, 10))) + "".join(map(str, r.integers(0, 10, 9)))

    def _time(self, hours=None, day=None) -> datetime:
        r = self.rng
        d = int(r.integers(0, self.window_days)) if day is None else day
        h = int(r.choice(24, p=HOUR_P)) if hours is None else int(r.choice(hours))
        return self.start + timedelta(days=d, hours=h, minutes=int(r.integers(0, 60)),
                                      seconds=int(r.integers(0, 60)))

    def _fill(self, t: str) -> str:
        r = self.rng
        return t.format(n6=int(r.integers(100000, 999999)), n4=int(r.integers(1000, 9999)),
                        amt=int(r.integers(100, 50000)), link="http://kyc-update.example/x")

    def _contact(self) -> str:
        return self.contacts[int(self.rng.choice(len(self.contacts), p=self.contact_p))]

    def _add(self, atype, ts, role, pattern, **fields):
        rows = self.rows[atype]
        rows.append(dict(case_id=self.case_id,
                         artifact_id=f"{self.case_id}-{atype}-{len(rows)}",
                         artifact_type=atype,
                         device="windows" if atype in ("web", "login") else "android",
                         source="live",
                         timestamp=ts.isoformat(),
                         role=role, pattern_id=pattern, **fields))

    # ------------------------------------------------------------- background
    def background(self, ranges: dict, scale: float) -> None:
        r = self.rng

        def n(key):
            lo, hi = ranges[key]
            return int(r.integers(int(lo * scale), int(hi * scale) + 1))

        for _ in range(n("sms")):
            u = r.random()
            if u < 0.55:                                   # personal chat
                direction = "out" if r.random() < 0.5 else "in"
                text = r.choice(PERSONAL_NOISE) if r.random() < 0.04 else r.choice(PERSONAL)
                self._add("sms", self._time(), "background", "", direction=direction,
                          counterparty=self._contact(), text=str(text))
            elif u < 0.70:                                 # work
                self._add("sms", self._time(), "background", "",
                          direction="out" if r.random() < 0.5 else "in",
                          counterparty=self._contact(), text=str(r.choice(WORK)))
            elif u < 0.88:                                 # promotions (incoming)
                self._add("sms", self._time(), "background", "", direction="in",
                          counterparty="VM-SHOPMRT", text=str(r.choice(PROMO)))
            else:                                          # bank messages (incoming)
                self._add("sms", self._time(), "background", "", direction="in",
                          counterparty="AD-BANKXX", text=self._fill(str(r.choice(BANK))))

        for _ in range(n("calls")):
            answered = r.random() < 0.85
            self._add("call", self._time(), "background", "",
                      direction="out" if r.random() < 0.5 else "in",
                      counterparty=self._contact(), answered=answered,
                      duration_s=int(r.lognormal(4.5, 0.8)) if answered else 0)

        for _ in range(n("web")):
            pool = BENIGN_DOMAINS + (TECH_DOMAINS if self.is_tech else [])
            self._add("web", self._time(), "background", "", domain=str(r.choice(pool)))
        if self.is_tech:                                   # security-curious developer: noise
            for _ in range(int(r.integers(0, 4))):
                self._add("web", self._time(), "background", "", domain="exploit-archive.example")

        for _ in range(n("logins")):
            country = "IN"
            if self.travels and r.random() < 0.3:
                country = "SG"
            self._add("login", self._time(), "background", "",
                      service=str(r.choice(SERVICES)), account=self.user,
                      success=bool(r.random() < 0.95), ip_country=country)

        for _ in range(n("transactions")):
            self._add("txn", self._time(), "background", "", direction="out",
                      counterparty=str(r.choice(self.payees)),
                      amount=float(round(r.lognormal(6.5, 1.0), 2)))
        # salary once a month + occasional refund from a one-off counterparty (noise)
        self._add("txn", self._time(day=0), "background", "", direction="in",
                  counterparty="employer-001", amount=float(round(r.uniform(2e4, 9e4), 2)))
        for _ in range(int(r.integers(0, 2))):
            self._add("txn", self._time(), "background", "", direction="in",
                      counterparty=f"merchant-refund-{r.integers(1000, 9999)}",
                      amount=float(round(r.uniform(100, 3000), 2)))

    # --------------------------------------------------------- planted patterns
    def plant(self, pattern: str, k: int, role: str) -> None:
        """Inject pattern `pattern` with intensity k. role = evidence | distractor."""
        r = self.rng
        if k <= 0:
            return
        if pattern == "F1_scam_sms":
            targets = [self._phone() for _ in range(max(1, k // 3))]
            for _ in range(k):
                self._add("sms", self._time(), role, pattern, direction="out",
                          counterparty=str(r.choice(targets)), text=self._fill(str(r.choice(SCAM))))
        elif pattern == "F2_mule_inflow":
            for _ in range(k):
                self._add("txn", self._time(), role, pattern, direction="in",
                          counterparty=f"acct-{r.integers(10**7, 10**8)}",
                          amount=float(round(r.uniform(2000, 50000), 2)))
        elif pattern == "F3_sms_blast":
            day, hour = int(r.integers(0, self.window_days)), int(r.integers(9, 21))
            base = self.start + timedelta(days=day, hours=hour)
            for _ in range(k):
                ts = base + timedelta(minutes=int(r.integers(0, 60)), seconds=int(r.integers(0, 60)))
                self._add("sms", ts, role, pattern, direction="out",
                          counterparty=self._phone(), text=str(r.choice(BLAST)))
        elif pattern == "H1_abusive_sms":
            victim = self.contacts[int(r.integers(0, len(self.contacts)))]
            for _ in range(k):
                self._add("sms", self._time(), role, pattern, direction="out",
                          counterparty=victim, text=str(r.choice(ABUSIVE)))
        elif pattern == "H2_repeat_calls":
            victim = self._phone()
            for _ in range(2 * k):                         # calls are cheaper -> 2x
                answered = r.random() < 0.25                # victim mostly rejects
                self._add("call", self._time(hours=range(9, 23)), role, pattern,
                          direction="out", counterparty=victim, answered=answered,
                          duration_s=int(r.integers(5, 25)) if answered else 0)
        elif pattern == "H3_night_calls":
            targets = [self._phone() for _ in range(int(r.integers(3, 6)))]
            for _ in range(k):
                self._add("call", self._time(hours=range(0, 5)), role, pattern,
                          direction="out", counterparty=str(r.choice(targets)),
                          answered=False, duration_s=0)
        elif pattern == "C1_hacktool_web":
            for _ in range(k):
                self._add("web", self._time(), role, pattern, domain=str(r.choice(HACK_DOMAINS)))
        elif pattern == "C2_cred_stuffing":
            for _ in range(int(1.5 * k)):
                self._add("login", self._time(), role, pattern, service=str(r.choice(SERVICES)),
                          account=f"victim{r.integers(10**5, 10**6)}@mailhub.example",
                          success=False, ip_country="IN")
        elif pattern == "C3_vpn_hopping":
            for _ in range(k):
                self._add("login", self._time(), role, pattern, service=str(r.choice(SERVICES)),
                          account=self.user, success=True,
                          ip_country=str(r.choice(COUNTRIES[1:])))
        else:
            raise ValueError(pattern)


def generate_case(case_id: str, label: str, tier: str, bcfg: dict, rng: np.random.Generator):
    """Return (rows_by_type, answer_key) for one case."""
    ts = bcfg["tier_settings"][tier]
    cb = CaseBuilder(case_id, rng, bcfg["window_days"])
    cb.background(bcfg["background"], ts["background_scale"])

    crime_classes = [c for c in CLASSES if c != "benign"]
    evidence, distractors = [], []

    if label == "benign":
        if rng.random() < bcfg["benign_distractor_prob"]:
            c = str(rng.choice(crime_classes))
            nlo, nhi = bcfg["benign_distractor_patterns"]
            n_p = int(rng.integers(nlo, nhi + 1))
            lo, hi = bcfg["benign_distractor_intensity"]
            for p in rng.choice(list(PATTERNS[c]), size=n_p, replace=False):
                cb.plant(str(p), int(rng.integers(lo, hi + 1)), "distractor")
                distractors.append(str(p))
    else:
        pats = list(PATTERNS[label])
        chosen = list(rng.choice(pats, size=ts["n_patterns"], replace=False))
        lo, hi = ts["intensity"]
        for p in chosen:
            cb.plant(str(p), int(rng.integers(lo, hi + 1)), "evidence")
            evidence.append(str(p))
        if rng.random() < ts["distractor_prob"]:
            other = str(rng.choice([c for c in crime_classes if c != label]))
            p = str(rng.choice(list(PATTERNS[other])))
            dlo, dhi = ts["distractor_intensity"]
            cb.plant(p, int(rng.integers(dlo, dhi + 1)), "distractor")
            distractors.append(p)

    key = dict(case_id=case_id, label=label, tier=tier,
               evidence_patterns=sorted(evidence),
               primary_features=sorted(PATTERN_TO_FEATURE[p] for p in evidence),
               distractor_patterns=sorted(distractors))
    return cb.rows, key
