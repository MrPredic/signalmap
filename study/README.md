# The sign of a label-free anomaly score is not identifiable

A detector fitted on healthy data alone fixes a threshold but never observes an
anomaly. Whether anomalies score **higher** or **lower** than healthy is
therefore an assumption about a distribution that was never seen — not
something learned from data.

We preregistered that claim, froze one recipe, and ran it over nine domains.
The assumption is wrong about as often as it is right.

| domain | n | AUC | 95% CI | direction | verdict |
|---|---|---|---|---|---|
| synth_neg | 80 | 0.1525 | [0.0763, 0.2412] | **inverted** | PASS |
| mimii_valve_id00 | 219 | 0.2786 | [0.2110, 0.3507] | **inverted** | PASS |
| mafaulda | 362 | 0.3124 | [0.2370, 0.3819] | inverted | REFUSED — null control failed |
| paderborn_kat_n15 | 120 | 0.3172 | [0.2235, 0.4194] | **inverted** | PASS |
| paderborn_kat | 480 | 0.4976 | [0.4434, 0.5513] | undetermined | REFUSED — null control failed |
| mimii_pump_id00 | 243 | 0.5203 | [0.4438, 0.5967] | undetermined | REFUSED |
| mimii_fan_id00 | 507 | 0.5609 | [0.4952, 0.6236] | undetermined | REFUSED |
| mimii_slider_id00 | 456 | 0.8254 | [0.7703, 0.8781] | **aligned** | PASS |
| synth_pos | 80 | 1.0000 | [1.0000, 1.0000] | **aligned** | PASS |

Same spec, same windowing, same aggregation, same seeds, every domain. An AUC
below 0.5 does not mean "fails to separate" — that would be a CI covering 0.5.
It means the ranking is **inverted**: anomalies are scored as more normal than
normal. Sign transfer between the decided domains is 0.400 over 20 ordered
pairs; knowing the sign in one domain tells you little about the next.

## This is not a new discovery

Stated up front, because a hostile review of the section below concluded the
theorem is trivial and known, and that verdict is correct. Without a specified
alternative there is no optimal test (Neyman–Pearson); one-class methods
define outlierness as distance from a boundary learned without outliers
(Schölkopf 2001, Tax & Duin 2004, Markou & Singh 2003); the limitation is
inventory in Chandola, Banerjee & Kumar (2009); and that anomalies can be
reconstructed well — and therefore score low — is established in the
autoencoder literature.

What is contributed here is narrower: the measurement across nine public
datasets under a single frozen, preregistered recipe with null controls, and
a detector that refuses rather than guesses. `THEORY.md` carries the full
prior-art note.

## Why this happens

The score is `max_j |φ_j(w) − m_j| / s_j` — a **distance** from the healthy
centre, where `m` and `s` come only from healthy data.

**Theorem.** Fix any healthy distribution `P₀` such that `S(X₀)` is atomless.
Then over choices of the anomaly distribution `P₁`, with the fitted detector
held completely fixed, `AUC(P₀, P₁)` attains every value in `[0, 1]`.

*Proof.* Put all of `P₁` at a point with `φ(w) = m`, so `S = 0` while
`S(X₀) > 0` almost surely: AUC = 0. Put it beyond the essential supremum of
`S(X₀)`: AUC = 1. Mixtures give everything between. ∎

The corollary is the practical part: `sign(AUC − ½)` is not a finding, it is an
assumption. And this is not a flaw in one implementation — **any** detector
that sees only healthy data and decides by distance from it inherits the gap.

Getting below 0.5 needs more than a shift. A shift in any direction raises the
distance. It requires the anomaly to sit *closer to the healthy centre than a
typical healthy window* — a contraction. `synth_neg` is built to do exactly
that and lands at AUC 0.1525, as registered before it was generated.

The same argument gives a closed form for one centred feature with dispersion
ratio `r`: `AUC(r) = (2/π)·arctan(r)`. Estimating `r` as a moment ratio and
comparing against the rank statistic is an independent route to the same
quantity: the **direction agrees in 4/4 domains whose direction the
measurement decided**, while magnitudes compress toward 0.5 (errors −0.183 to
+0.284) exactly as the derivation's stated limits predict.

## Verify it yourself, offline, without trusting us

Every claim above ships as an Ed25519-signed receipt. The verifier is a single
file that **imports nothing from signalmap** — stdlib plus `cryptography`:

```bash
python tools/verify_receipt.py study/receipts/sign_mimii_valve_id00.receipt.json
# PASS — verdict PASS, integrity only (NOT authenticity)

# pin the key for authenticity, then try to change a number:
python - <<'EOF'
import json; p="study/receipts/sign_mimii_valve_id00.receipt.json"
d=json.load(open(p)); d["evidence"]["auc"]=0.9
json.dump(d, open("/tmp/tampered.json","w"))
EOF
python tools/verify_receipt.py /tmp/tampered.json --pubkey <hex>
# FAIL — signature does not match receipt body
```

`study/manifests/` carries the sha256 of every one of the 7623 recordings, so
a reviewer can rebuild the banks from the public sources and confirm they got
the same bytes.

## What we are *not* claiming

- **Five of thirteen preregistered domains were never obtained** (`cwru`,
  `ecn`, `geomag`, `battery_eis`, `volcano` — see `PREREG.md` amendment 5 for
  the counts that disqualified each). No substitute domain was added.
- **Two domains failed their own null control and are excluded**, not quietly
  counted. Sorting MAFAULDA's healthy files sorts them by rotation speed, and
  Paderborn's by operating point, so each healthy set separates from itself
  before any fault is involved. `paderborn_kat_n15` restricts to one operating
  point and then passes — that is why both appear in the table.
- **H2 is weak.** Only 6 of 9 domains clear their own shuffle-null 95th
  percentile by more than 0.02; `mimii_fan_id00` sits exactly on it and is
  counted as undecided, not as support.
- **The preregistered H3 was withdrawn**, not reported: it compared a
  window-calibrated threshold against a recording mean and was structurally
  silent. Amendment 3 records that the invalid numbers had already been seen
  when the error was found. The corrected measurement is labelled post-hoc.
- `PREREG.md` carries **ten dated amendments**, several of which record our own
  mistakes, including one that corrects a wrong causal attribution we had
  already published in an earlier amendment.

## What it cost us

Measured on this project's own detector, across the same nine domains: the
shipped alarm's gap between anomalies and normals is **not above zero in eight
of them**, and in six it **never fires at all** — including
`paderborn_kat_n15`, whose ranking is CI-fest inverted. Ranking quality and
shipped behaviour are not the same claim, and only the first one was ever
advertised.

## What changed in the product

The detector no longer assumes the direction:

```python
det = DistilledDetector.fit(spec, healthy_windows)   # direction is None
det.decide(w).verdict                                 # 'REFUSED'

v = det.calibrate_direction(anchor_windows, labels, groups=recording_ids)
v.sign, v.ci_lo, v.ci_hi        # +1 or -1, only when the CI clears 0.5
det.decide(w).verdict           # now 'ALARM' or 'QUIET'
```

`groups` makes the bootstrap resample **recordings**: twenty windows cut from
one signal are one observation, and treating them as twenty lets chance
structure inside a single recording look like a direction.

Two further fixes came out of the same work: the standalone verifier was
hardened against 23 concrete attacks on the gap between the bytes that are
signed and the fields a human reads (duplicate JSON keys, `NaN`/`Infinity`
literals, integers past 2⁵³, unpaired surrogates, non-UTF-8 input), and a
guard floor that could let a feature which is constant *by construction*
dominate the score was made relative to the feature's own magnitude.

## Reproduce

```bash
study/tools/fetch_remaining_domains.sh          # public sources, ~4 GB
python study/tools/make_mimii_sign_domains.py   # build banks to the contract
python study/tools/check_domain_bank.py data/signdomains/<domain>   # ACCEPTED
python study/tools/sign_identifiability_readout.py --refresh
```

Seeds are fixed at 0 throughout; the readout caches per domain and regenerates
the report and the receipts.

- `PREREG.md` — the preregistration, frozen before any bank was built, with
  all ten amendments in the order they happened (German original, preserved
  verbatim as the audit trail).
- `THEORY.md` — the non-identifiability argument in full, including which
  steps are exact and which need an assumption real data may violate.
- `BANK_CONTRACT.md` — the rules every domain bank had to satisfy, including
  the rule that whoever builds a bank may never compute a separation metric.
- `REPORT*.md` — generated outputs: the main table, the alarm measurement, the
  theory check, and the probe that ruled out a numerical explanation for the
  first inversion we saw.

### A note on the internal paths you will see

`PREREG.md`, `THEORY.md`, `BANK_CONTRACT.md`, the `REPORT*.md` files, the
receipts and the manifests are reproduced **byte for byte** from the working
tree that produced them, so they still name internal paths such as
`research/factory/…`. That is deliberate and it is checkable:

```bash
python - <<'EOF'
import json, hashlib
ov = json.load(open("study/receipts/sign_identifiability_overall.receipt.json"))
print(ov["input_hashes"]["report_sha256"])
print(hashlib.sha256(open("study/REPORT.md","rb").read()).hexdigest())
EOF
```

The overall receipt commits to the sha256 of `REPORT.md`, and each per-domain
receipt commits to the sha256 of its manifest. Tidying the paths would have
been cosmetic and would have broken every one of those hashes. The runnable
copies under `study/tools/` are the ones with public paths.
