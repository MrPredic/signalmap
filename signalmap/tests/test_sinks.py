"""Built-in sinks: the pluggable surface users extend, and the persistence
contract `signalmap run --sink parquet` depends on."""
import socket

import numpy as np
import pytest

from signalmap.core import Result, available
from signalmap.frame import Frame
from signalmap.sinks import ParquetSink, QdrantSink, QuestDBSink, StdoutSink

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _result(seq=0, node_id=1, score=0.5, dim=32, energy=12.5):
    frame = Frame(is_spectrum=False, node_id=node_id, seq=seq, ts_us=seq * 1000,
                  sr_hz=16_000, n=512, payload=np.zeros(512, dtype=np.int16))
    return Result(frame=frame, feature=np.zeros(256, dtype=np.float32),
                  embedding=np.arange(dim, dtype=np.float32), score=score,
                  meta={"node_id": node_id, "seq": seq, "sensor_class": 0,
                        "energy_rms": energy})


def _dead_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_all_builtin_sinks_are_discoverable_by_name():
    assert {"stdout", "parquet", "questdb", "qdrant"} <= set(available("sink"))


def test_stdout_sink_honours_the_every_n_downsample(capsys):
    s = StdoutSink(every=3)
    for i in range(9):
        s.emit(_result(seq=i))
    s.close()
    out = capsys.readouterr().out
    assert out.count("node=") == 3
    assert "9 frames processed" in out


def test_stdout_sink_survives_a_result_without_energy(capsys):
    r = _result()
    del r.meta["energy_rms"]
    StdoutSink().emit(r)                       # must not raise
    assert "e_rms" not in capsys.readouterr().out


def test_parquet_sink_round_trips_embeddings_bit_exactly(tmp_path):
    path = tmp_path / "nested" / "emb.parquet"   # nested: dir must be created
    s = ParquetSink(str(path))
    for i in range(5):
        s.emit(_result(seq=i, score=i / 10))
    s.close()

    t = pq.read_table(str(path))
    assert t.num_rows == 5
    assert t.column("seq").to_pylist() == list(range(5))
    assert t.column("score").to_pylist() == [i / 10 for i in range(5)]
    back = np.frombuffer(t.column("embedding").to_pylist()[0], dtype=np.float32)
    np.testing.assert_array_equal(back, np.arange(32, dtype=np.float32))


def test_parquet_sink_writes_nothing_when_no_frames_arrived(tmp_path):
    path = tmp_path / "empty.parquet"
    ParquetSink(str(path)).close()
    assert not path.exists()


def test_questdb_sink_offline_swallows_emits(capsys):
    s = QuestDBSink(host="127.0.0.1", port=_dead_port())
    s.emit(_result())                          # must not raise
    s.close()
    assert "offline" in capsys.readouterr().out


def test_qdrant_sink_offline_swallows_emits(capsys):
    s = QdrantSink()
    s.q.client = None                          # force the degraded path
    s.emit(_result())                          # must not raise
    s.close()


def test_qdrant_point_ids_stay_unique_across_nodes_and_wrapping_seqs():
    ids = set()
    s = QdrantSink()
    s.q.client = None
    for node in (1, 2):
        for seq in (0, 1, 2**32 - 1):
            r = _result(seq=seq, node_id=node)
            ids.add((r.meta["node_id"] << 32) | (r.meta["seq"] & 0xFFFF_FFFF))
    assert len(ids) == 6
