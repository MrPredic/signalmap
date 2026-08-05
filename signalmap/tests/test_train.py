"""Training path: `signalmap train` must produce an artifact the runtime loads back."""
import numpy as np
import pytest
import torch

from signalmap import train as train_mod
from signalmap.model import SpectralAutoencoder

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SR = 16_000


def _write_parquet(path, n_rows=4, n_samples=512):
    rows = []
    for i in range(n_rows):
        t = np.arange(n_samples) / SR
        sig = 300 * np.sin(2 * np.pi * (200 + 50 * i) * t)
        rows.append(np.clip(sig, -2048, 2047).astype("<i2").tobytes())
    table = pa.table({
        "label": [f"mat{i}" for i in range(n_rows)],
        "sr_hz": [SR] * n_rows,
        "samples": rows,
    })
    pq.write_table(table, path)
    return path


def test_synthetic_dataset_is_deterministic_and_shaped_for_the_model():
    a = train_mod.synthetic_dataset(6)
    b = train_mod.synthetic_dataset(6)
    assert a.shape == (6, train_mod.N_BINS)
    np.testing.assert_allclose(a, b)


def test_load_dataset_reads_recorded_frames_into_features(tmp_path):
    feats = train_mod.load_dataset(str(_write_parquet(tmp_path / "ds.parquet")))
    assert feats.shape == (4, train_mod.N_BINS)
    assert np.isfinite(feats).all()


def test_train_writes_a_checkpoint_the_runtime_can_load_back(tmp_path):
    out = tmp_path / "nested" / "model.pt"     # nested: dir must be created for us
    train_mod.train(train_mod.synthetic_dataset(8), epochs=1, out=str(out))
    assert out.exists()

    model = SpectralAutoencoder(n_bins=train_mod.N_BINS, latent_dim=32)
    model.load_state_dict(torch.load(str(out), map_location="cpu", weights_only=True))


def test_training_reduces_reconstruction_loss(tmp_path):
    feats = train_mod.synthetic_dataset(32)
    x = torch.from_numpy(feats).float()

    def loss_of(path):
        m = SpectralAutoencoder(n_bins=train_mod.N_BINS, latent_dim=32)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            recon, _ = m(x)
        return float(((recon - x) ** 2).mean())

    # Both runs must start from the SAME random init, or the comparison is
    # between two different models and fails at random (observed: 1 in 3
    # parallel runs). Seeded, "more epochs -> lower loss" is a real claim.
    def train_seeded(epochs, name):
        torch.manual_seed(0)
        train_mod.train(feats, epochs=epochs, out=str(tmp_path / name))

    train_seeded(1, "a.pt")
    train_seeded(15, "b.pt")
    assert loss_of(str(tmp_path / "b.pt")) < loss_of(str(tmp_path / "a.pt"))


def test_cli_without_a_data_source_fails_with_an_actionable_message(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["signalmap-train"])
    with pytest.raises(SystemExit) as exc:
        train_mod.main()
    assert "--dataset" in str(exc.value) and "--synthetic" in str(exc.value)


def test_cli_rejects_zero_synthetic_frames_instead_of_silently_asking_for_a_source(monkeypatch):
    # `--synthetic 0` is a user error about the COUNT; reporting "provide a
    # source" sends them looking in the wrong place.
    monkeypatch.setattr("sys.argv", ["signalmap-train", "--synthetic", "0"])
    with pytest.raises(SystemExit) as exc:
        train_mod.main()
    assert "0" in str(exc.value) or "at least" in str(exc.value).lower()
