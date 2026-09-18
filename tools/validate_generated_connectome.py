#!/usr/bin/env python3
"""Independent structural validation for FlyBrain V1.02 generated data."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

TARGET = 16669
MAGIC = b"FBC102\x00\x00"
NODE_SIZE = 25
EDGE_SIZE = 12
HEADER_SIZE = 16


def main(root: Path) -> None:
    raw = root / "app" / "src" / "main" / "res" / "raw"
    bin_path = raw / "malecns_reduced.bin"
    report_path = raw / "malecns_reduced_report.json"
    meta_path = root / "app" / "src" / "main" / "java" / "com" / "example" / "flybrain" / "GeneratedConnectomeMeta.kt"
    main_path = root / "app" / "src" / "main" / "java" / "com" / "example" / "flybrain" / "MainActivity.kt"

    assert bin_path.is_file(), bin_path
    assert report_path.is_file(), report_path
    assert meta_path.is_file(), meta_path
    assert main_path.is_file(), main_path

    report = json.loads(report_path.read_text())
    data = bin_path.read_bytes()
    meta = meta_path.read_text()
    main_text = main_path.read_text()

    assert report["dataset"] == "MaleCNS v1.0"
    assert report["flybrain_version"] == "1.02"
    assert report["binary_format"] == "FBC102"
    assert report["node_record_bytes"] == NODE_SIZE
    assert report["edge_record_bytes"] == EDGE_SIZE
    assert report["neurons_retained"] == TARGET
    assert report["edges_retained"] > 0
    assert report["retained_sensor_to_desc_edges"] > 0
    assert report["retained_desc_to_motor_edges"] > 0
    assert report["retained_sensor_to_desc_contacts"] > 0
    assert report["retained_desc_to_motor_contacts"] > 0

    assert data[:8] == MAGIC, data[:8]
    n, e = struct.unpack_from("<II", data, 8)
    assert n == TARGET, n
    assert e == report["edges_retained"], (e, report["edges_retained"])

    expected = HEADER_SIZE + n * NODE_SIZE + e * EDGE_SIZE
    assert len(data) == expected, (len(data), expected)

    node_start = HEADER_SIZE
    edge_start = node_start + n * NODE_SIZE

    route_nonzero = [0, 0, 0]
    motor_nodes = 0
    desc_nodes = 0

    for i in range(n):
        off = node_start + i * NODE_SIZE
        body, superclass, side, channel, motor_role, desc_role, rf, rt, re = struct.unpack_from(
            "<qbbbbbfff", data, off
        )
        assert body > 0
        assert -128 <= superclass <= 127
        assert side in (-1, 0, 1)
        assert 0 <= channel <= 4
        assert 0 <= motor_role <= 7
        assert 0 <= desc_role <= 4
        for x in (rf, rt, re):
            assert math.isfinite(x)
            assert 0.0 <= x <= 1.0
        route_nonzero[0] += rf > 0
        route_nonzero[1] += rt > 0
        route_nonzero[2] += re > 0
        motor_nodes += motor_role > 0
        desc_nodes += desc_role > 0

    for i in range(e):
        off = edge_start + i * EDGE_SIZE
        src, dst, weight = struct.unpack_from("<iif", data, off)
        assert 0 <= src < n, (i, src)
        assert 0 <= dst < n, (i, dst)
        assert math.isfinite(weight), (i, weight)
        assert weight != 0.0, i

    assert edge_start + e * EDGE_SIZE == len(data)

    assert "FORMAT_MAGIC = \"FBC102\"" in meta
    assert "FORMAT_VERSION = 102" in meta
    assert "NEURONS = 16669" in meta
    assert "connectomeLoaded" in main_text
    assert "expectedBytes" in main_text
    assert "bytes.size.toLong() != expectedBytes" in main_text
    assert "buildFallbackBrain" not in main_text
    assert "plasticityEnabled = false" in main_text

    print("FBC102 independent validation: OK")
    print(f"neurons={n}")
    print(f"edges={e}")
    print(f"bytes={len(data)}")
    print(f"route_forward_nonzero={route_nonzero[0]}")
    print(f"route_turn_nonzero={route_nonzero[1]}")
    print(f"route_escape_nonzero={route_nonzero[2]}")
    print(f"motor_role_nodes={motor_nodes}")
    print(f"descending_role_nodes={desc_nodes}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
