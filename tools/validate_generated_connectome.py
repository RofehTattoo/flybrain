#!/usr/bin/env python3
"""Independent structural validation for FlyBrain V1.09 generated data."""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

TARGET = 16669
MAGIC = b"FBC102\x00\x00"  # Binary format remains FBC102; app release is V1.09.
NODE_SIZE = 25
EDGE_SIZE = 12
HEADER_SIZE = 16


def parse_meta_int(meta: str, name: str) -> int:
    m = re.search(rf"const val {re.escape(name)} = (-?\d+)", meta)
    assert m, f"missing metadata constant {name}"
    return int(m.group(1))


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
    assert report["flybrain_version"] == "1.09"
    assert report["binary_format"] == "FBC102"
    assert report["node_record_bytes"] == NODE_SIZE
    assert report["edge_record_bytes"] == EDGE_SIZE
    assert report["neurons_retained"] == TARGET
    assert report["edges_retained"] > 0
    assert report["contacts_retained_signed"] > 0

    # A direct DN->MN edge is diagnostic only. The required motor architecture
    # is the real two-hop DN -> intermediate/premotor -> MN path.
    assert report["retained_sensor_to_desc_edges"] > 0
    assert report["retained_sensor_to_desc_contacts"] > 0
    sensor_desc = report["retained_sensor_to_desc_by_channel_role_contacts"]
    assert len(sensor_desc) == 4
    assert all(len(row) == 5 for row in sensor_desc)
    assert sensor_desc[0][4] > 0, "no retained visual -> escape-DN contacts"
    assert report["retained_desc_to_intermediate_edges"] > 0
    assert report["retained_intermediate_to_motor_edges"] > 0
    assert report["retained_desc_to_intermediate_to_motor_paths"] > 0

    two_hop = report["retained_desc_to_intermediate_to_motor_by_role_paths"]
    assert len(two_hop) == 5
    assert all(len(row) == 8 for row in two_hop)
    assert sum(map(sum, two_hop)) > 0

    role_counts = report["descending_role_counts"]
    assert int(role_counts.get("1", 0)) > 0
    assert int(role_counts.get("2", 0)) > 0
    assert int(role_counts.get("4", 0)) > 0

    route_counts_report = report["route_score_selected_counts"]
    assert route_counts_report["forward_nonzero"] > 0
    assert route_counts_report["turn_nonzero"] > 0
    assert route_counts_report["escape_nonzero"] > 0

    ranges = report["channel_ranges"]
    names = ("visual", "olfactory", "gustatory", "mechanosensory", "other")
    previous_end = 0
    for name in names:
        pair = ranges[name]
        assert len(pair) == 2, (name, pair)
        start, end = map(int, pair)
        assert 0 <= start <= end <= TARGET, (name, pair)
        assert start == previous_end, (name, pair, previous_end)
        previous_end = end
    assert previous_end == TARGET

    # channel_ranges describes sensory classification; population_ranges describes
    # the actual contiguous runtime blocks consumed by MainActivity.
    population_ranges = report["population_ranges"]
    population_names = ("visual", "olfactory", "gustatory", "mechanosensory", "descending", "ascending", "motor", "other")
    previous_end = 0
    for name in population_names:
        pair = population_ranges[name]
        assert len(pair) == 2, (name, pair)
        start, end = map(int, pair)
        assert 0 <= start <= end <= TARGET, (name, pair)
        assert start == previous_end, (name, pair, previous_end)
        previous_end = end
    assert previous_end == TARGET

    # Generated Kotlin metadata must exactly mirror the runtime population
    # ranges. This prevents channel=4 (all non-sensory neurons) from being
    # mistaken for the final OTHER runtime block.
    meta_expected = {
        "VIS_START": population_ranges["visual"][0], "VIS_END": population_ranges["visual"][1],
        "OLF_START": population_ranges["olfactory"][0], "OLF_END": population_ranges["olfactory"][1],
        "GUST_START": population_ranges["gustatory"][0], "GUST_END": population_ranges["gustatory"][1],
        "MECH_START": population_ranges["mechanosensory"][0], "MECH_END": population_ranges["mechanosensory"][1],
        "DESC_START": population_ranges["descending"][0], "DESC_END": population_ranges["descending"][1],
        "ASC_START": population_ranges["ascending"][0], "ASC_END": population_ranges["ascending"][1],
        "VMOTOR_START": population_ranges["motor"][0], "VMOTOR_END": population_ranges["motor"][1],
        "OTHER_START": population_ranges["other"][0], "OTHER_END": population_ranges["other"][1],
    }
    for name, value in meta_expected.items():
        assert parse_meta_int(meta, name) == int(value), (name, value)

    assert parse_meta_int(meta, "NEURONS") == TARGET
    assert parse_meta_int(meta, "FORMAT_VERSION") == 102

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
    channels = []

    for i in range(n):
        off = node_start + i * NODE_SIZE
        body, superclass, side, channel, motor_role, desc_role, rf, rt, re_ = struct.unpack_from(
            "<qbbbbbfff", data, off
        )
        assert body > 0
        assert -128 <= superclass <= 127
        assert side in (-1, 0, 1)
        assert 0 <= channel <= 4
        assert 0 <= motor_role <= 7
        assert 0 <= desc_role <= 4
        for x in (rf, rt, re_):
            assert math.isfinite(x)
            assert 0.0 <= x <= 1.0
        route_nonzero[0] += rf > 0
        route_nonzero[1] += rt > 0
        route_nonzero[2] += re_ > 0
        motor_nodes += motor_role > 0
        desc_nodes += desc_role > 0
        channels.append(channel)

    assert route_nonzero == [
        int(route_counts_report["forward_nonzero"]),
        int(route_counts_report["turn_nonzero"]),
        int(route_counts_report["escape_nonzero"]),
    ]
    assert all(route_nonzero)
    assert motor_nodes > 0
    assert desc_nodes > 0

    # Channel blocks must be exactly the generated contiguous ranges.
    for code, name in enumerate(names):
        start, end = ranges[name]
        block = channels[start:end]
        assert len(block) == end - start
        assert all(c == code for c in block), (name, code)
    assert channels == sorted(channels)

    edge_pairs = set()
    previous = None
    for i in range(e):
        off = edge_start + i * EDGE_SIZE
        src, dst, weight = struct.unpack_from("<iif", data, off)
        assert 0 <= src < n, (i, src)
        assert 0 <= dst < n, (i, dst)
        assert math.isfinite(weight), (i, weight)
        assert weight != 0.0, i
        key = (dst, src)
        assert previous is None or previous <= key, (i, previous, key)
        previous = key
        edge_pairs.add((src, dst))

    assert edge_start + e * EDGE_SIZE == len(data)
    assert len(edge_pairs) == e, "duplicate edge records detected"

    assert 'FORMAT_MAGIC = "FBC102"' in meta
    assert 'FORMAT_VERSION = 102' in meta
    assert 'NEURONS = 16669' in meta
    assert "connectomeLoaded" in main_text
    assert "expectedBytes" in main_text
    assert "bytes.size.toLong() != expectedBytes" in main_text
    assert "buildFallbackBrain" not in main_text
    assert "plasticityEnabled = false" in main_text

    print("FBC102 independent validation for FlyBrain V1.09: OK")
    print(f"neurons={n}")
    print(f"edges={e}")
    print(f"bytes={len(data)}")
    print(f"route_forward_nonzero={route_nonzero[0]}")
    print(f"route_turn_nonzero={route_nonzero[1]}")
    print(f"route_escape_nonzero={route_nonzero[2]}")
    print(f"motor_role_nodes={motor_nodes}")
    print(f"descending_role_nodes={desc_nodes}")
    print(f"two_hop_paths={report['retained_desc_to_intermediate_to_motor_paths']}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
