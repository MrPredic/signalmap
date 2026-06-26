# Contributing

SignalMap grows by plugins. Adding one is small and self-contained.

## Add a Source (new signal origin)
```python
# signalmap/sources.py (or your own module imported by the CLI)
from .core import register
from .frame import Frame

@register("source", "mysensor")
class MySensor:
    def frames(self):
        while True:
            samples = ...           # np.int16 array, RAW (no filtering!)
            yield Frame(False, node_id=1, seq=n, ts_us=t,
                        sr_hz=8000, n=len(samples), payload=samples,
                        sensor_class=42)   # metadata only
```
Then: `signalmap run --source mysensor --sink stdout`.

## Add a Transform / Model / Sink
Same pattern — satisfy the matching `Protocol` in `signalmap/core.py` and
`@register("transform"|"model"|"sink", "name")`. Run `signalmap plugins` to
confirm it registered.

## Rules
1. **Respect the bias contract** (see `docs/ARCHITECTURE.md`). No silent
   filtering, no labels into the model, raw energy preserved, gaps reported.
2. **No new required dependencies** in core. Heavy deps go in
   `pyproject.toml` extras and are lazy-imported inside the plugin.
3. **Tests pass**: `pytest -q`. Add a test for new framing/DSP logic.
4. **Claims stay honest.** Don't describe an anomaly score as a "discovery".

## Dev setup
```bash
pip install -e .[all]
pytest -q
```

By contributing you agree your work is licensed under Apache-2.0.
