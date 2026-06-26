"""Universal-platform proof — simulation only, no hardware.

Demonstrates that ONE unsupervised pipeline works across 8 heterogeneous sensor
domains. Builds a multi-domain dataset, trains the Conv-AE blind to sensor
class, then measures two things:

  1. Cross-domain anomaly discovery — do injected 'novel effect' frames rank as
     top outliers regardless of which domain they hide in?
  2. Physics self-organization — does the latent space cluster by sensor domain
     on its own? Measured by nearest-neighbour class agreement (the class labels
     are used ONLY here, for verification — never in training).

    python -m backend.simulate_universal
"""
from __future__ import annotations

import numpy as np
import torch

from .dsp import raw_to_features
from .embed import Embedder
from .model import SpectralAutoencoder
from .synth import SensorClass, build_dataset

N_BINS = 256
DATASET = "data/universal.parquet"
MODEL = "artifacts/model_universal.pt"


def _load():
    import pyarrow.parquet as pq

    t = pq.read_table(DATASET)
    return (
        t.column("label").to_pylist(),
        np.array(t.column("sensor_class").to_pylist()),
        t.column("sr_hz").to_pylist(),
        t.column("samples").to_pylist(),
    )


def _features(blobs, srs):
    feats = []
    for b, sr in zip(blobs, srs):
        s = np.frombuffer(b, dtype="<i2").astype(np.float32)
        feats.append(raw_to_features(s, sr, n_bins=N_BINS).mag)
    return np.stack(feats)


def main() -> None:
    rows = build_dataset(DATASET, per_class=60, anomalies=8)
    labels, classes, srs, blobs = _load()
    print(f"dataset: {rows} frames across 8 sensor domains + injected anomalies\n")

    feats = _features(blobs, srs)
    is_anom = np.array([l.startswith("ANOMALY") for l in labels])

    # --- train on KNOWN sensor data only; anomalies are unseen (realistic) ---
    x = torch.from_numpy(feats[~is_anom]).float()
    model = SpectralAutoencoder(n_bins=N_BINS, latent_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    for ep in range(40):
        for (b,) in loader:
            opt.zero_grad()
            recon, _ = model(b)
            loss = torch.mean((recon - b) ** 2)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), MODEL)

    # --- embed everything; collect latent vectors + raw energy ---
    embedder = Embedder(model)
    vecs, energy = [], []
    for i in range(len(feats)):
        s = np.frombuffer(blobs[i], dtype="<i2").astype(np.float32)
        feat = raw_to_features(s, srs[i], n_bins=N_BINS)
        emb, _ez = embedder.embed(feat, node_id=0, seq=i, ts_us=0)
        vecs.append(emb.vector)
        energy.append(feat.energy_rms)
    vecs = np.stack(vecs); energy = np.array(energy)

    # latent novelty = cosine distance to the KNOWN-data manifold (k nearest
    # normal points, self excluded). Far from everything learned = novel.
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    ref = vn[~is_anom]
    sim_ref = vn @ ref.T                      # (all, normals)
    sim_ref[sim_ref > 0.99999] = -1.0         # drop self-match for normals
    knn = np.sort(sim_ref, axis=1)[:, -8:]    # 8 nearest known points
    novelty = 1.0 - knn.mean(axis=1)

    def _z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    # combine z-scored novelty + z-scored raw energy (the payoff signal)
    scores = _z(novelty) + _z(energy)

    # --- metric 1: cross-domain anomaly discovery ---
    k = int(is_anom.sum())
    top_k = set(np.argsort(scores)[-k:].tolist())
    caught = sum(is_anom[i] for i in top_k)
    print(f"[1] Cross-domain anomaly discovery: {caught}/{k} injected novel "
          f"effects in top-{k} by score")
    hit_domains = {SensorClass(classes[i]).name for i in np.where(is_anom)[0]}
    print(f"    anomalies were hidden across domains: {sorted(hit_domains)}\n")

    # --- metric 2: physics self-organization (NN class agreement) ---
    v = vecs[~is_anom]
    c = classes[~is_anom]
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    sim = vn @ vn.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(axis=1)
    agree = float(np.mean(c[nn] == c))
    print(f"[2] Physics self-organization (unsupervised): nearest-neighbour "
          f"sensor-class agreement = {agree:.1%}")
    print("    (model never saw the class — high % means the latent space "
          "organized by physics on its own)\n")

    # per-domain purity
    print("    per-domain NN purity:")
    for cls in sorted(set(c)):
        m = c == cls
        print(f"      {SensorClass(cls).name:11s} {np.mean(c[nn][m] == cls):.0%}")

    # --- render the universal landscape HTML ---
    _render(vecs, scores, classes, labels, is_anom, "artifacts/universal_map.html")
    print("\n-> artifacts/universal_map.html (latent landscape, colored by domain)")


def _render(vecs, scores, classes, labels, is_anom, out):
    # PCA via SVD (no sklearn)
    xm = vecs - vecs.mean(axis=0)
    _u, _s, vt = np.linalg.svd(xm, full_matrices=False)
    coords = xm @ vt[:2].T
    pts = [
        {
            "x": float(coords[i, 0]), "y": float(coords[i, 1]),
            "cls": int(classes[i]), "name": labels[i],
            "score": float(scores[i]), "anom": bool(is_anom[i]),
        }
        for i in range(len(scores))
    ]
    import json

    from .visualize import _HTML  # reuse styling shell

    # universal map needs class coloring -> small dedicated script
    palette = ["#ff7a1a", "#1aa3ff", "#7CFC00", "#ff4dd2", "#ffd11a",
               "#ff3b3b", "#1affd1", "#b07cff"]
    html = _UNIVERSAL_HTML.replace("/*DATA*/", json.dumps(pts)) \
                          .replace("/*PALETTE*/", json.dumps(palette))
    with open(out, "w") as f:
        f.write(html)


_UNIVERSAL_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>SignalMap — Universal Sensor Landscape</title>
<style>
 body{margin:0;background:#0a0e1a;color:#e6ecff;font:14px system-ui,sans-serif}
 header{padding:16px 22px;border-bottom:1px solid #1c2540}
 h1{margin:0;font-size:18px}h1 span{color:#ff7a1a}
 .sub{color:#8a97c2;font-size:12px;margin-top:4px}
 svg{width:100vw;height:calc(100vh - 116px)}
 #tip{position:fixed;pointer-events:none;background:#111a33;border:1px solid #2a3666;
   padding:7px 9px;border-radius:6px;font-size:12px;opacity:0}
 #leg{padding:8px 22px;font-size:12px;color:#aeb9e0}
 #leg b{color:#fff}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px}
</style></head><body>
<header><h1>Signal<span>Map</span> — Universal Sensor Landscape</h1>
<div class="sub">One unsupervised model, 8 sensor domains. Color = sensor domain (verification only) · orange ring = discovered novel-effect anomaly. The model was blind to domain — separation is self-organized.</div></header>
<div id="leg"></div><svg id="plot"></svg><div id="tip"></div>
<script>
const data=/*DATA*/, pal=/*PALETTE*/;
const names=['ACOUSTIC','VIBRATION','OPTICAL','RF','THERMAL','TRIBO','MAGNETIC','CAPACITIVE'];
const leg=document.getElementById('leg');
names.forEach((n,i)=>leg.innerHTML+=`<span class="sw" style="background:${pal[i]}"></span>${n}`);
const svg=document.getElementById('plot'),tip=document.getElementById('tip');
const W=svg.clientWidth,H=svg.clientHeight,pad=46;
const xs=data.map(d=>d.x),ys=data.map(d=>d.y);
const xn=Math.min(...xs),xx=Math.max(...xs),yn=Math.min(...ys),yx=Math.max(...ys);
const sx=v=>pad+(v-xn)/(xx-xn||1)*(W-2*pad),sy=v=>pad+(v-yn)/(yx-yn||1)*(H-2*pad);
const NS='http://www.w3.org/2000/svg';
data.forEach(d=>{const c=document.createElementNS(NS,'circle');
 c.setAttribute('cx',sx(d.x));c.setAttribute('cy',sy(d.y));
 c.setAttribute('r',d.anom?8:4.5);
 c.setAttribute('fill',d.anom?'#fff':pal[d.cls%8]);c.setAttribute('fill-opacity',.85);
 if(d.anom){c.setAttribute('stroke','#ff7a1a');c.setAttribute('stroke-width',3);}
 c.onmousemove=e=>{tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';
   tip.style.top=(e.clientY+12)+'px';
   tip.innerHTML=`<b>${d.name}</b><br>domain ${names[d.cls]||d.cls}<br>score ${d.score.toFixed(4)}`
     +(d.anom?'<br><span style="color:#ff7a1a">⚑ novel effect</span>':'');};
 c.onmouseleave=()=>tip.style.opacity=0;svg.appendChild(c);});
</script></body></html>"""


if __name__ == "__main__":
    main()
