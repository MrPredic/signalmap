"""The pitch asset: a self-contained HTML latent map.

Takes a recorded dataset + trained model, computes embeddings, projects to 2D
(PCA via numpy SVD — zero extra deps; UMAP used automatically if installed),
and writes a single standalone HTML file with an SVG scatter. Points are
colored by anomaly score; the top outliers are ringed. Drop this into a pitch
deck as a screenshot/GIF.

    python -m backend.visualize --dataset data/dataset.parquet \
        --model artifacts/model.pt --out artifacts/latent_map.html
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .dsp import raw_to_features
from .embed import Embedder, anomaly_score
from .model import SpectralAutoencoder

N_BINS = 256


def _load(path: str):
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    return (
        t.column("label").to_pylist(),
        t.column("sr_hz").to_pylist(),
        t.column("samples").to_pylist(),
    )


def _project_2d(vectors: np.ndarray) -> np.ndarray:
    try:
        import umap

        return umap.UMAP(n_components=2, n_neighbors=min(15, len(vectors) - 1)).fit_transform(vectors)
    except Exception:
        # PCA via SVD, no sklearn needed.
        x = vectors - vectors.mean(axis=0)
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:2].T


def build(dataset: str, model_path: str, out: str) -> None:
    labels, srs, blobs = _load(dataset)
    model = SpectralAutoencoder(n_bins=N_BINS, latent_dim=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    embedder = Embedder(model)

    vecs, scores, recon, energy = [], [], [], []
    for i, (raw_bytes, sr) in enumerate(zip(blobs, srs)):
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
        feat = raw_to_features(samples, sr, n_bins=N_BINS)
        emb, energy_z = embedder.embed(feat, node_id=0, seq=i, ts_us=0)
        vecs.append(emb.vector)
        recon.append(emb.recon_error)
        energy.append(emb.energy_rms)
        scores.append(anomaly_score(emb.recon_error, knn_distance=1.0, energy_z=energy_z))

    vecs = np.stack(vecs)
    coords = _project_2d(vecs)
    scores = np.array(scores)
    top = set(np.argsort(scores)[-max(1, len(scores) // 20):].tolist())  # top 5%

    points = [
        {
            "x": float(coords[i, 0]), "y": float(coords[i, 1]),
            "label": labels[i], "score": float(scores[i]),
            "recon": float(recon[i]), "energy": float(energy[i]),
            "anomaly": i in top,
        }
        for i in range(len(scores))
    ]
    _write_html(points, out)
    n_anom = sum(p["anomaly"] for p in points)
    print(f"  {len(points)} points, {n_anom} flagged anomalies -> {out}")


def _write_html(points: list[dict], out: str) -> None:
    html = _HTML.replace("/*DATA*/", json.dumps(points))
    with open(out, "w") as f:
        f.write(html)


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>SignalMap — Latent Material Landscape</title>
<style>
 body{margin:0;background:#0a0e1a;color:#e6ecff;font:14px/1.5 system-ui,sans-serif}
 header{padding:18px 24px;border-bottom:1px solid #1c2540}
 h1{margin:0;font-size:18px;letter-spacing:.3px}
 h1 span{color:#ff7a1a}
 .sub{color:#8a97c2;font-size:12px;margin-top:4px}
 #wrap{display:flex}
 svg{flex:1;height:calc(100vh - 70px)}
 circle{cursor:pointer;transition:r .1s}
 .anom{stroke:#ff7a1a;stroke-width:2.5}
 #tip{position:fixed;pointer-events:none;background:#111a33;border:1px solid #2a3666;
      padding:8px 10px;border-radius:6px;font-size:12px;opacity:0;max-width:240px}
 #legend{position:fixed;right:18px;bottom:18px;background:#111a33aa;padding:10px 12px;
      border-radius:8px;border:1px solid #1c2540;font-size:12px}
</style></head><body>
<header><h1>Signal<span>Map</span> — Latent Material Landscape</h1>
<div class="sub">Unsupervised embeddings of raw signal frames. Color = anomaly score · orange ring = high-performance outlier. No DIN/ISO labels used.</div></header>
<div id="wrap"><svg id="plot"></svg></div>
<div id="tip"></div>
<div id="legend">● low score &nbsp; ● high score &nbsp; <span style="color:#ff7a1a">◯ anomaly</span></div>
<script>
const data = /*DATA*/;
const svg = document.getElementById('plot'), tip = document.getElementById('tip');
const W = svg.clientWidth, H = svg.clientHeight, pad = 40;
const xs = data.map(d=>d.x), ys = data.map(d=>d.y), ss = data.map(d=>d.score);
const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
const smax=Math.max(...ss),smin=Math.min(...ss);
const sx=v=>pad+(v-xmin)/(xmax-xmin||1)*(W-2*pad);
const sy=v=>pad+(v-ymin)/(ymax-ymin||1)*(H-2*pad);
function color(s){const t=(s-smin)/(smax-smin||1);
  const r=Math.round(40+t*215),g=Math.round(120-t*60),b=Math.round(255-t*180);
  return `rgb(${r},${g},${b})`;}
const NS='http://www.w3.org/2000/svg';
data.forEach(d=>{
  const c=document.createElementNS(NS,'circle');
  c.setAttribute('cx',sx(d.x));c.setAttribute('cy',sy(d.y));
  c.setAttribute('r',d.anomaly?7:4);c.setAttribute('fill',color(d.score));
  c.setAttribute('fill-opacity',.85);if(d.anomaly)c.setAttribute('class','anom');
  c.addEventListener('mousemove',e=>{tip.style.opacity=1;
    tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';
    tip.innerHTML=`<b>${d.label}</b><br>score ${d.score.toFixed(4)}<br>`+
      `recon ${d.recon.toFixed(4)} · energy ${d.energy.toFixed(0)}`+
      (d.anomaly?'<br><span style="color:#ff7a1a">⚑ outlier</span>':'');});
  c.addEventListener('mouseleave',()=>tip.style.opacity=0);
  svg.appendChild(c);
});
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="Render latent-space HTML map")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="artifacts/model.pt")
    p.add_argument("--out", default="artifacts/latent_map.html")
    args = p.parse_args()
    build(args.dataset, args.model, args.out)


if __name__ == "__main__":
    main()
