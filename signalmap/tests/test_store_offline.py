"""Both stores promise to degrade to no-ops when the server is down.

That promise is what lets the demo run on a laptop with no Docker stack, so it
is worth a test that actually points them at a dead port.
"""
import socket

import numpy as np

from signalmap.store import QdrantNovelty, QuestDBWriter


def _dead_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()          # nothing is listening here now
    return port


def test_questdb_writer_offline_is_a_silent_no_op(capsys):
    w = QuestDBWriter(host="127.0.0.1", port=_dead_port())
    assert w.sock is None
    w.write_measurement(node_id=1, ts_us=1, energy_rms=0.5,
                        recon_error=0.1, anomaly_score=0.3)   # must not raise
    assert "offline" in capsys.readouterr().out


def test_questdb_writer_sends_line_protocol_when_a_server_listens():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        w = QuestDBWriter(host="127.0.0.1", port=srv.getsockname()[1])
        assert w.sock is not None
        w.write_measurement(node_id=3, ts_us=1_000, energy_rms=0.5,
                            recon_error=0.25, anomaly_score=2.0)
        conn, _ = srv.accept()
        line = conn.recv(4096).decode()
        conn.close()
    finally:
        srv.close()

    assert line.startswith("signal,node=3 ")
    assert "energy_rms=0.5" in line and "anomaly_score=2.0" in line
    assert line.rstrip().endswith("1000000")     # us -> ns, no float notation
    assert line.endswith("\n")


def test_questdb_writer_drops_the_socket_after_a_failed_write(capsys):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    w = QuestDBWriter(host="127.0.0.1", port=srv.getsockname()[1])
    conn, _ = srv.accept()
    conn.close()
    srv.close()
    w.sock.close()          # force the next sendall to fail

    w.write_measurement(1, 1, 0.5, 0.1, 0.3)    # must not raise
    assert w.sock is None                        # and must not retry a dead socket
    w.write_measurement(1, 2, 0.5, 0.1, 0.3)    # still a no-op afterwards


def test_qdrant_novelty_offline_reports_maximal_novelty(capsys):
    q = QdrantNovelty(host="127.0.0.1", port=_dead_port())
    assert q.client is None
    assert q.novelty(np.zeros(32)) == 1.0
    q.upsert(1, np.zeros(32), {"label": "x"})    # must not raise
    assert "offline" in capsys.readouterr().out
