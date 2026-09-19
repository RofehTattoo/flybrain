#!/usr/bin/env python3
"""Build a deterministic 16,669-neuron MaleCNS v1.0 reduction for FlyBrain V1.06.

The reduction is derived from the published MaleCNS v1.0 annotation and weighted
connectivity tables. It keeps exactly 10% of the 166,691-neuron census by
stratifying on the published superclass and selecting the highest-connectivity
cells within each superclass. Only published edges between retained neurons
are embedded; no graph edge is invented here.
"""
from __future__ import annotations

import json
import math
import re
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
TARGET = 16669
FORMAT_MAGIC = b"FBC102\x00\x00"
FORMAT_VERSION = 102
NODE_SIZE = 25
EDGE_SIZE = 12
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

NT_SIGN = {
    "acetylcholine": 1.0,
    "ach": 1.0,
    "gaba": -1.0,
    "gamma-aminobutyric acid": -1.0,
    # Modeling convention used by recent Drosophila VNC connectome simulations.
    # Receptor-specific glutamatergic effects are not represented in this reduced model.
    "glutamate": -1.0,
    "glutamatergic": -1.0,
}

SUPERCLASS_CODE = {}

def download(path: Path, name: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    url = BASE + name
    tmp = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading {name} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def clean(x):
    if pd.isna(x):
        return ""
    return str(x)


def classify_channel(row) -> int:
    """0 visual, 1 olfactory, 2 gustatory, 3 mechanosensory/proprioceptive, 4 other."""
    sc = clean(row.get("superclass", ""))
    cl = clean(row.get("class", "")).lower()
    sub = clean(row.get("subclass", "")).lower()
    typ = clean(row.get("type", "")).lower()
    inst = clean(row.get("instance", "")).lower()
    name = clean(row.get("name", "")).lower()
    text = " ".join((cl, sub, typ, inst, name))
    if sc in {"visual_projection", "visual_centrifugal"} or "visual" in text or "optic lobe" in text:
        return 0
    if sc == "ol_sensory" or "olf" in text or "antennal lobe" in text:
        return 1
    if "gust" in text or "taste" in text:
        return 2
    if sc in {"vnc_sensory", "sensory_ascending", "sensory_descending"}:
        return 3
    if sc.startswith("cb_sensory"):
        return 3
    if any(k in text for k in (
        "mechanosensory", "proprio", "bristle", "hair plate",
        "campaniform", "chordotonal", "johnston",
    )):
        return 3
    return 4


def classify_motor_role(row) -> int:
    """Map curated VNC motor annotations to a body output category.
    0=non-motor, 1=leg, 2=wing, 3=haltere, 4=neck, 5=abdomen, 6=jump, 7=other.
    This uses published annotation text, never the runtime neuron index.
    """
    sc = clean(row.get("superclass", ""))
    if sc != "vnc_motor":
        return 0
    fields = []
    for c in ("type", "instance", "name", "class", "subclass", "primary_neuropil", "nerve", "target", "muscle", "annotation", "group"):
        if c in row.index:
            fields.append(clean(row.get(c, "")))
    text = " ".join(fields).lower()
    if any(k in text for k in ("tergotrochanteral", "jump", "ttmn")):
        return 6
    if any(k in text for k in ("haltere", "halter")):
        return 3
    if any(k in text for k in ("wing", "dvm", "dlm", "flight", "steering")):
        return 2
    if any(k in text for k in ("neck", "cervical")):
        return 4
    if any(k in text for k in ("abdominal", "abdomen", "abdominal")):
        return 5
    if any(k in text for k in ("leg", "t1", "t2", "t3", "coxa", "femur", "tibia", "tars", "trochanter", "levator", "depressor", "flexor", "extensor")):
        return 1
    return 7


def classify_descending_role(row) -> int:
    """Assign a descriptive motor-family label to a descending neuron.

    The label is metadata only: it never creates, removes, or rewrites an edge.
    V1.04 uses explicit published cell-type names for the few descending
    populations whose behavioural role is established, and only then falls
    back to annotation keywords. This avoids the previous-version failure mode where
    almost all 1,314 DNs were left as role 0 simply because the annotation
    did not literally contain words such as "walk" or "escape".

    0=generic/unknown, 1=forward/walking, 2=turn/steering,
    3=backward/halting, 4=fast escape.
    """
    sc = clean(row.get("superclass", ""))
    if sc != "descending_neuron":
        return 0

    fields = []
    for c in ("type", "instance", "name", "class", "subclass", "primary_neuropil", "target", "annotation", "group"):
        if c in row.index:
            fields.append(clean(row.get(c, "")))
    text = " ".join(fields).lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)

    # Established walking/forward DNs.
    if any(k in compact for k in ("dng100", "dng97", "dnb08")) or any(
        k in text for k in ("forward walking", "walking command")
    ):
        return 1

    # Established steering/turning DNs and major central-complex descending
    # targets described for goal-directed steering.
    if any(k in compact for k in (
        "dna01", "dna02", "dna03", "dna04", "dna11",
        "dnb01", "dng13", "dng13a",
    )) or any(k in text for k in (
        "turn", "steer", "turning", "steering", "pfl3",
    )):
        return 2

    # Moonwalker/backward and halting/landing pathways.
    if any(k in compact for k in (
        "dnp50", "mdn", "moonwalker", "dnp07", "dnp09",
    )) or any(k in text for k in (
        "backward walking", "backward locomotion", "halting",
    )):
        return 3

    # DNp01 is the Giant Fiber descending neuron and drives the fast escape
    # pathway downstream of optic-lobe looming detectors.
    if "dnp01" in compact or any(k in text for k in (
        "giant fiber", "giant-fiber", "giant fibre", "giant-fibre",
    )):
        return 4

    return 0

def main(root: Path) -> None:
    raw = root / "build" / "malecns_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for key, name in FILES.items():
        download(raw / name, name)

    annotations = pd.read_feather(raw / FILES["annotations"])
    if "status" in annotations.columns:
        traced = annotations[annotations["status"].astype(str).str.lower().eq("traced")].copy()
    else:
        traced = annotations.copy()
    traced = traced[traced["superclass"].notna()].copy()
    traced["bodyId"] = traced["bodyId"].astype(np.int64)
    traced = traced.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)
    if len(traced) < TARGET:
        raise RuntimeError(f"Only {len(traced)} traced annotated neurons available; expected >= {TARGET}")

    ids = traced["bodyId"].to_numpy(np.int64)
    degree = np.zeros(len(ids), dtype=np.float64)

    # One streaming pass over the 1.1 GB weighted graph to measure connectivity.
    weights_path = raw / FILES["weights"]
    reader = ipc.open_file(pa.memory_map(str(weights_path), "r"))
    for bi in range(reader.num_record_batches):
        b = reader.get_batch(bi)
        pre = b.column(b.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False).astype(np.int64)
        post = b.column(b.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False).astype(np.int64)
        w = b.column(b.schema.get_field_index("weight")).to_numpy(zero_copy_only=False).astype(np.float64)
        pi = np.searchsorted(ids, pre)
        po = np.searchsorted(ids, post)
        pv = (pi < len(ids)) & (ids[np.minimum(pi, len(ids)-1)] == pre)
        qv = (po < len(ids)) & (ids[np.minimum(po, len(ids)-1)] == post)
        if pv.any(): np.add.at(degree, pi[pv], w[pv])
        if qv.any(): np.add.at(degree, po[qv], w[qv])
        if bi % 50 == 0:
            print(f"degree pass batch {bi}/{reader.num_record_batches}", flush=True)

    traced["degree"] = degree
    counts = traced.groupby("superclass", sort=True).size().to_dict()
    total = len(traced)

    # V1.04 functional-route analysis. The reduction now preferentially preserves
    # measured two-hop pathways in the published graph instead of using a single
    # aggregate bridge score. For each candidate neuron we estimate whether the
    # full connectome contains paths of the form sensor -> candidate -> DN and
    # DN -> candidate -> VNC motor neuron, separately for forward, turning and
    # escape-related routes. No edge is invented by this analysis.
    traced["channel"] = traced.apply(classify_channel, axis=1)
    traced["motor_role"] = traced.apply(classify_motor_role, axis=1)
    traced["descending_role"] = traced.apply(classify_descending_role, axis=1)

    is_sensory = traced["channel"].to_numpy(np.int8) < 4
    is_desc = traced["superclass"].astype(str).eq("descending_neuron").to_numpy()
    is_motor = traced["superclass"].astype(str).eq("vnc_motor").to_numpy()
    channel = traced["channel"].to_numpy(np.int8)
    desc_role = traced["descending_role"].to_numpy(np.int8)
    motor_role = traced["motor_role"].to_numpy(np.int8)

    role_counts_source = np.bincount(desc_role[is_desc], minlength=5)
    missing_roles = [
        name for name, role in (("forward", 1), ("turn", 2), ("escape", 4))
        if int(role_counts_source[role]) == 0
    ]
    if missing_roles:
        raise RuntimeError(
            "Published MaleCNS annotations did not expose required DN role(s): "
            + ", ".join(missing_roles)
        )

    # Incoming sensory weight per candidate, separated by modality.
    sensor_in = np.zeros((4, len(ids)), dtype=np.float64)
    # Candidate -> DN role weight.
    cell_to_desc = np.zeros((5, len(ids)), dtype=np.float64)
    # DN role -> candidate weight.
    desc_to_cell = np.zeros((5, len(ids)), dtype=np.float64)
    # Candidate -> motor-role weight.
    cell_to_motor = np.zeros((8, len(ids)), dtype=np.float64)

    direct_sensor_desc = np.zeros((4, 5), dtype=np.float64)
    direct_desc_motor = np.zeros((5, 8), dtype=np.float64)

    for bi in range(reader.num_record_batches):
        b = reader.get_batch(bi)
        pre = b.column(b.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False).astype(np.int64)
        post = b.column(b.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False).astype(np.int64)
        w = b.column(b.schema.get_field_index("weight")).to_numpy(zero_copy_only=False).astype(np.float64)
        pi = np.searchsorted(ids, pre)
        po = np.searchsorted(ids, post)
        pv = (pi < len(ids)) & (ids[np.minimum(pi, len(ids)-1)] == pre)
        qv = (po < len(ids)) & (ids[np.minimum(po, len(ids)-1)] == post)
        both = pv & qv & (w > 0)
        if not both.any():
            continue
        ai = pi[both]
        ci = po[both]
        wd = w[both]

        # Sensor -> candidate.
        m = is_sensory[ai]
        if m.any():
            for ch in range(4):
                mm = m & (channel[ai] == ch)
                if mm.any():
                    np.add.at(sensor_in[ch], ci[mm], wd[mm])

        # Candidate -> descending neuron.
        m = is_desc[ci]
        if m.any():
            for role in range(1, 5):
                mm = m & (desc_role[ci] == role)
                if mm.any():
                    np.add.at(cell_to_desc[role], ai[mm], wd[mm])

        # Descending neuron -> candidate.
        m = is_desc[ai]
        if m.any():
            for role in range(1, 5):
                mm = m & (desc_role[ai] == role)
                if mm.any():
                    np.add.at(desc_to_cell[role], ci[mm], wd[mm])

        # Candidate -> motor neuron.
        m = is_motor[ci]
        if m.any():
            for mr in range(1, 8):
                mm = m & (motor_role[ci] == mr)
                if mm.any():
                    np.add.at(cell_to_motor[mr], ai[mm], wd[mm])

        # Direct route counts are retained for the build report.
        m = is_sensory[ai] & is_desc[ci]
        if m.any():
            np.add.at(
                direct_sensor_desc,
                (channel[ai[m]], desc_role[ci[m]]),
                wd[m],
            )
        m = is_desc[ai] & is_motor[ci]
        if m.any():
            np.add.at(
                direct_desc_motor,
                (desc_role[ai[m]], motor_role[ci[m]]),
                wd[m],
            )

        if bi % 50 == 0:
            print(f"functional route pass batch {bi}/{reader.num_record_batches}", flush=True)

    # Two-hop route scores. Geometric means prevent a cell with only one strong
    # side of a route from dominating. The scores are used for neuron selection
    # and are also embedded as descriptive runtime metadata; they never alter an edge.
    # Route-family scores preserve both sides of the circuit:
    #   sensor -> DN   and   DN -> premotor -> motor.
    # A candidate does not need to be the same cell on both sides. This is
    # important for real layered circuits such as LC4/LPLC2 -> DNp01 -> VNC.
    # The scores are still purely descriptive and never create an edge.
    sensory_forward = np.sqrt(
        np.maximum(0.0, sensor_in.sum(axis=0) * cell_to_desc[1])
    )
    visual_turn = np.sqrt(np.maximum(0.0, sensor_in[0] * cell_to_desc[2]))
    threat_escape = np.sqrt(
        np.maximum(0.0, (sensor_in[0] + sensor_in[3]) * cell_to_desc[4])
    )

    forward_motor_path = np.sqrt(
        np.maximum(0.0, desc_to_cell[1] * cell_to_motor[1])
    )
    # Turning/steering can recruit coordinated motor outputs.
    turn_motor_outputs = cell_to_motor[1:].sum(axis=0)
    turn_motor_path = np.sqrt(
        np.maximum(0.0, desc_to_cell[2] * turn_motor_outputs)
    )
    escape_motor_path = np.sqrt(np.maximum(
        0.0,
        desc_to_cell[4] * (cell_to_motor[1] + cell_to_motor[2] + cell_to_motor[6])
    ))

    route_forward = sensory_forward + forward_motor_path
    route_turn = visual_turn + turn_motor_path
    route_escape = threat_escape + escape_motor_path
    route_sensorimotor = np.sqrt(np.maximum(
        0.0,
        sensor_in.sum(axis=0) * cell_to_motor[1:].sum(axis=0)
    ))

    # V1.04 is fail-closed at the source-analysis level. A build is not allowed
    # to publish a reduced graph whose three intended behavioural route families
    # are silently absent. These are measured topology scores, not synthetic
    # currents or generated edges.
    missing_routes = [
        name for name, score in (
            ("forward", route_forward),
            ("turn", route_turn),
            ("escape", route_escape),
        )
        if not np.any(score > 0)
    ]
    if missing_routes:
        raise RuntimeError(
            "MaleCNS v1.0 route analysis produced no positive measured "
            + ", ".join(missing_routes)
            + " route candidate(s); refusing to publish V1.04."
        )

    def normalize_score(x):
        m = float(np.nanmax(x)) if len(x) else 0.0
        return (x / m) if m > 0 else np.zeros_like(x)

    route_forward_n = normalize_score(route_forward)
    route_turn_n = normalize_score(route_turn)
    route_escape_n = normalize_score(route_escape)
    route_sensorimotor_n = normalize_score(route_sensorimotor)
    degree_n = normalize_score(traced["degree"].to_numpy(np.float64))

    traced["route_forward"] = route_forward_n
    traced["route_turn"] = route_turn_n
    traced["route_escape"] = route_escape_n
    traced["route_sensorimotor"] = route_sensorimotor_n
    traced["route_score"] = (
        0.32 * route_forward_n
        + 0.25 * route_turn_n
        + 0.38 * route_escape_n
        + 0.18 * route_sensorimotor_n
    )

    # Selection strategy for V1.04:
    # 1) retain every curated descending neuron and every curated VNC motor neuron;
    # 2) retain at least one representative per published neuron type;
    # 3) reserve a substantial quota for measured two-hop functional routes;
    # 4) fill remaining slots proportionally by superclass, ranked by route score
    #    and then degree. This explicitly protects intermediate premotor cells.
    forced = traced[traced["superclass"].astype(str).isin({"descending_neuron", "vnc_motor"})].copy()

    type_col = "type" if "type" in traced.columns else None
    if type_col is None:
        traced["type"] = traced["bodyId"].astype(str)
        type_col = "type"
    traced[type_col] = traced[type_col].fillna("").astype(str)

    type_rep = (
        traced.sort_values(["degree", "bodyId"], ascending=[False, True])
        .drop_duplicates(["superclass", type_col], keep="first")
    )

    seed = pd.concat([forced, type_rep], ignore_index=True).drop_duplicates("bodyId")

    if len(seed) > TARGET:
        forced_ids = set(forced.bodyId.astype(int).tolist())
        keep_forced = forced.drop_duplicates("bodyId")
        optional_types = type_rep[~type_rep.bodyId.isin(forced_ids)].sort_values(
            ["route_score", "degree", "bodyId"], ascending=[False, False, True]
        )
        seed = pd.concat(
            [keep_forced, optional_types.head(max(0, TARGET - len(keep_forced)))],
            ignore_index=True,
        ).drop_duplicates("bodyId")

    remaining_slots = TARGET - len(seed)
    if remaining_slots < 0:
        raise AssertionError(("seed exceeds target", len(seed), TARGET))

    seed_ids = set(seed.bodyId.astype(int).tolist())
    pool = traced[~traced.bodyId.isin(seed_ids)].copy()

    # Route preservation is intended to protect intermediate circuit cells,
    # not to spend the route quota on sensory/DN/MN populations already handled
    # by the forced/type-diversity stages. Ascending neurons remain eligible as
    # measured intermediate/feedback cells.
    # Route-preservation pool includes sensory bridge cells as well as central
    # intermediates. This is important for canonical pathways such as
    # visual/looming -> LC4/LPLC2 -> DNp01 -> VNC. The motor-path validator
    # below still requires its two-hop bridge cell itself to be non-sensory.
    intermediate_pool = pool[
        ~pool["superclass"].astype(str).isin({"descending_neuron", "vnc_motor"})
    ].copy()

    # Reserve route cells by functional family. Quotas are capped by the
    # available slots, so the exact 16,669 target is always respected.
    route_quota = {
        "route_escape": 900,
        "route_forward": 650,
        "route_turn": 600,
        "route_sensorimotor": 450,
    }
    route_parts = []
    route_ids = set()

    # Explicitly protect cells with a positive measured turn route before
    # allocating the other route-family quotas. No edge is created here.
    positive_turn = intermediate_pool[
        intermediate_pool["route_turn"] > 0
    ].sort_values(
        ["route_turn", "route_score", "degree", "bodyId"],
        ascending=[False, False, False, True],
    )

    if len(positive_turn) and remaining_slots > 0:
        take_turn = min(
            route_quota["route_turn"],
            remaining_slots,
            len(positive_turn),
        )
        turn_seed = positive_turn.head(take_turn)
        route_parts.append(turn_seed)
        route_ids.update(turn_seed.bodyId.astype(int).tolist())
        remaining_slots -= len(turn_seed)

    for score_col, quota in route_quota.items():
        if score_col == "route_turn":
            continue
        if remaining_slots <= 0:
            break
        take = min(quota, remaining_slots)
        candidates = intermediate_pool[~intermediate_pool.bodyId.isin(route_ids)].sort_values(
            [score_col, "route_score", "degree", "bodyId"],
            ascending=[False, False, False, True],
        )
        part = candidates.head(take)
        if len(part):
            route_parts.append(part)
            route_ids.update(part.bodyId.astype(int).tolist())
            remaining_slots -= len(part)

    if route_parts:
        route_seed = pd.concat(route_parts, ignore_index=True)
        seed = pd.concat([seed, route_seed], ignore_index=True).drop_duplicates("bodyId")

    remaining_slots = TARGET - len(seed)
    if remaining_slots < 0:
        raise AssertionError(("route seed exceeds target", len(seed), TARGET))

    seed_ids = set(seed.bodyId.astype(int).tolist())
    pool = traced[~traced.bodyId.isin(seed_ids)].copy()

    pool_counts = pool.groupby("superclass", sort=True).size().to_dict()
    raw_extra = {sc: n * remaining_slots / max(1, len(pool)) for sc, n in pool_counts.items()}
    extra_quota = {sc: int(math.floor(q)) for sc, q in raw_extra.items()}

    while sum(extra_quota.values()) < remaining_slots:
        candidates = [sc for sc in pool_counts if extra_quota[sc] < pool_counts[sc]]
        if not candidates:
            break
        sc = max(candidates, key=lambda x: (raw_extra[x] - extra_quota[x], x))
        extra_quota[sc] += 1

    while sum(extra_quota.values()) > remaining_slots:
        sc = max(extra_quota, key=lambda x: (extra_quota[x], x))
        extra_quota[sc] -= 1

    extras = []
    for sc, group in pool.groupby("superclass", sort=True):
        n = min(extra_quota.get(sc, 0), len(group))
        if n:
            extras.append(
                group.sort_values(
                    ["route_score", "degree", "bodyId"],
                    ascending=[False, False, True],
                ).head(n)
            )

    selected = pd.concat([seed] + extras, ignore_index=True).drop_duplicates("bodyId")

    if len(selected) < TARGET:
        selected_ids_now = set(selected.bodyId.astype(int).tolist())
        extra_pool = traced[~traced.bodyId.isin(selected_ids_now)]
        selected = pd.concat(
            [
                selected,
                extra_pool.sort_values(
                    ["route_score", "degree", "bodyId"],
                    ascending=[False, False, True],
                ).head(TARGET - len(selected)),
            ],
            ignore_index=True,
        )

    if len(selected) > TARGET:
        forced_ids = set(forced.bodyId.astype(int).tolist())
        keep_forced = selected[selected.bodyId.isin(forced_ids)]
        optional = selected[~selected.bodyId.isin(forced_ids)].sort_values(
            ["route_score", "degree", "bodyId"], ascending=[False, False, True]
        )
        selected = pd.concat(
            [keep_forced, optional.head(max(0, TARGET - len(keep_forced)))],
            ignore_index=True,
        )

    selected = selected.drop_duplicates("bodyId").reset_index(drop=True)
    if len(selected) != TARGET:
        raise AssertionError((len(selected), TARGET))

    for route_name in ("route_forward", "route_turn", "route_escape"):
        if not np.any(selected[route_name].to_numpy(np.float64) > 0):
            raise AssertionError(
                f"selected set contains no non-zero measured {route_name} cell"
            )

    # Stable anatomical ordering: sensory channels first, then descending,
    # ascending, motor and the remaining central/intrinsic populations. This
    # gives the Android runtime contiguous anatomical readout blocks.
    selected["channel"] = selected.apply(classify_channel, axis=1)
    selected["motor_role"] = selected.apply(classify_motor_role, axis=1)
    selected["descending_role"] = selected.apply(classify_descending_role, axis=1)
    def block(row):
        sc = str(row["superclass"])
        ch = int(row["channel"])
        if ch < 4:
            return ch
        if sc == "descending_neuron":
            return 4
        if sc == "ascending_neuron":
            return 5
        if sc == "vnc_motor":
            return 6
        return 7
    selected["block"] = selected.apply(block, axis=1)
    selected = selected.sort_values(["block", "superclass", "bodyId"]).reset_index(drop=True)
    selected_ids = selected["bodyId"].to_numpy(np.int64)
    index = {int(b): i for i, b in enumerate(selected_ids)}

    # Materialize all per-selected-neuron metadata arrays before the edge pass.
    # These arrays must be indexed in the same stable order as `selected` and
    # are consumed below when classifying retained sensor->DN and DN->motor
    # edges and when writing route metadata to the binary file.
    channel_selected = selected["channel"].to_numpy(np.int8)
    descending_role_selected = selected["descending_role"].to_numpy(np.int8)
    motor_role_selected = selected["motor_role"].to_numpy(np.int8)
    route_forward_selected = selected["route_forward"].to_numpy(np.float64)
    route_turn_selected = selected["route_turn"].to_numpy(np.float64)
    route_escape_selected = selected["route_escape"].to_numpy(np.float64)

    # Neurotransmitter predictions provide the sign model used by the simulator.
    nt = pd.read_feather(raw / FILES["neurotransmitters"])
    nt_cols = [c for c in ["body", "bodyId"] if c in nt.columns]
    nt_id_col = nt_cols[0] if nt_cols else None
    nt["_id"] = nt[nt_id_col].astype(np.int64) if nt_id_col else -1
    nt_name = None
    for c in ["consensus_nt", "predicted_nt", "celltype_predicted_nt"]:
        if c in nt.columns:
            nt_name = c
            break
    nt_map = dict(zip(nt["_id"].tolist(), nt[nt_name].astype(str).str.strip().str.lower().tolist())) if nt_name else {}

    # Second streaming pass: retain only published edges between selected neurons.
    edges = []
    contacts = 0
    candidate_edges = 0
    unresolved_edges = 0
    retained_sensor_desc = np.zeros((4, 5), dtype=np.int64)
    retained_desc_motor = np.zeros((5, 8), dtype=np.int64)
    retained_sensor_desc_edges = 0
    retained_desc_motor_edges = 0

    # Retained two-hop motor routes. Direct DN->MN contacts are not required:
    # in the biological VNC, descending influence commonly reaches motor neurons
    # through premotor/intermediate neurons. Count only paths formed by published
    # edges between neurons that actually survived the 16,669-node reduction.
    retained_desc_to_intermediate_edges = 0
    retained_intermediate_to_motor_edges = 0
    retained_desc_to_intermediate_to_motor_paths = 0
    retained_two_hop_by_role = np.zeros((5, 8), dtype=np.int64)
    desc_to_intermediate = {}
    intermediate_to_motor = {}
    for bi in range(reader.num_record_batches):
        b = reader.get_batch(bi)
        pre = b.column(b.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False).astype(np.int64)
        post = b.column(b.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False).astype(np.int64)
        w = b.column(b.schema.get_field_index("weight")).to_numpy(zero_copy_only=False).astype(np.int64)
        mask = np.isin(pre, selected_ids) & np.isin(post, selected_ids) & (w > 0)
        if mask.any():
            for a, c, d in zip(pre[mask], post[mask], w[mask]):
                candidate_edges += 1
                sign = NT_SIGN.get(nt_map.get(int(a), ""), 0.0)
                if sign == 0.0:
                    unresolved_edges += 1
                    continue
                src_i = index[int(a)]
                dst_i = index[int(c)]
                edges.append((src_i, dst_i, float(d) * sign))
                contacts += int(d)
                src_ch = int(channel_selected[src_i])
                dst_desc = int(descending_role_selected[dst_i])
                src_desc = int(descending_role_selected[src_i])
                dst_mr = int(motor_role_selected[dst_i])
                if src_ch < 4 and dst_desc > 0:
                    retained_sensor_desc[src_ch, dst_desc] += int(d)
                    retained_sensor_desc_edges += 1
                if src_desc > 0 and dst_mr > 0:
                    retained_desc_motor[src_desc, dst_mr] += int(d)
                    retained_desc_motor_edges += 1

                # Intermediate/premotor cells are non-sensory, non-DN,
                # non-MN retained neurons. This avoids counting a sensory
                # feedback cell as the sole "premotor" bridge.
                src_is_intermediate = (
                    src_desc == 0
                    and int(motor_role_selected[src_i]) == 0
                    and int(channel_selected[src_i]) >= 4
                )
                dst_is_intermediate = (
                    dst_desc == 0
                    and dst_mr == 0
                    and int(channel_selected[dst_i]) >= 4
                )
                if src_desc > 0 and dst_is_intermediate:
                    desc_to_intermediate.setdefault(src_i, set()).add(dst_i)
                    retained_desc_to_intermediate_edges += 1
                if src_is_intermediate and dst_mr > 0:
                    intermediate_to_motor.setdefault(src_i, set()).add(dst_i)
                    retained_intermediate_to_motor_edges += 1
        if bi % 50 == 0:
            print(f"edge pass batch {bi}/{reader.num_record_batches}", flush=True)

    if retained_sensor_desc[0, 4] <= 0:
        raise RuntimeError(
            "No retained visual -> escape-DN contacts survived the reduction; "
            "refusing to publish V1.04."
        )

    # Count actual retained DN -> intermediate -> motor paths. Each path is a
    # real two-edge path in the published graph; no synthetic bridge is added.
    for dn_i, mids in desc_to_intermediate.items():
        dn_role = int(descending_role_selected[dn_i])
        if dn_role <= 0:
            continue
        for mid_i in mids:
            motors = intermediate_to_motor.get(mid_i)
            if not motors:
                continue
            retained_desc_to_intermediate_to_motor_paths += len(motors)
            for motor_i in motors:
                mr = int(motor_role_selected[motor_i])
                if mr > 0:
                    retained_two_hop_by_role[dn_role, mr] += 1

    target_totals = np.zeros(TARGET, dtype=np.float64)
    for src, dst, raw_weight in edges:
        target_totals[dst] += abs(raw_weight)
    normalized_edges = []
    for src, dst, raw_weight in edges:
        denom = max(1.0, target_totals[dst])
        normalized_edges.append((src, dst, float(raw_weight) / denom * 0.42))
    edges = normalized_edges
    edges.sort(key=lambda e: (e[1], e[0]))

    ranges = {}
    for code, name in [(0,"visual"),(1,"olfactory"),(2,"gustatory"),(3,"mechanosensory"),(4,"other")]:
        idxs = np.where(channel_selected == code)[0]
        ranges[name] = [int(idxs.min()), int(idxs.max()+1)] if len(idxs) else [0,0]

    sc_list = sorted(selected["superclass"].astype(str).unique().tolist())
    SUPERCLASS_CODE.clear()
    SUPERCLASS_CODE.update({s:i for i,s in enumerate(sc_list)})

    # Compact binary FBC102:
    # 8-byte magic + node/edge counts + 25-byte node records + 12-byte edges.
    # The three float route scores are descriptive metadata computed from
    # published two-hop topology; they do not create or modify edges.
    out = root / "app" / "src" / "main" / "res" / "raw" / "malecns_reduced.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(FORMAT_MAGIC)
        f.write(struct.pack("<II", TARGET, len(edges)))
        for i, row in selected.iterrows():
            body = int(row.bodyId)
            sc = SUPERCLASS_CODE[str(row.superclass)]
            side_text = clean(row.get("somaSide", ""))
            side = 1 if side_text == "R" else (-1 if side_text == "L" else 0)
            ch = int(channel_selected[i])
            f.write(struct.pack(
                "<qbbbbbfff",
                body,
                sc,
                side,
                ch,
                int(row.motor_role),
                int(row.descending_role),
                float(route_forward_selected[i]),
                float(route_turn_selected[i]),
                float(route_escape_selected[i]),
            ))
        for src, dst, weight in edges:
            f.write(struct.pack("<iif", int(src), int(dst), float(weight)))

    def block_range(block_id):
        idxs = np.where(selected["block"].to_numpy(np.int8) == block_id)[0]
        return (int(idxs.min()), int(idxs.max()+1)) if len(idxs) else (0,0)

    # Runtime population ranges follow the stable anatomical block order.
    # channel=4 means "non-sensory" and spans DESC/ASC/MOTOR/OTHER; it is
    # therefore not the runtime OTHER population.
    desc = block_range(4)
    asc = block_range(5)
    vmotor = block_range(6)
    other = block_range(7)
    population_ranges = {
        "visual": block_range(0), "olfactory": block_range(1),
        "gustatory": block_range(2), "mechanosensory": block_range(3),
        "descending": desc, "ascending": asc, "motor": vmotor, "other": other,
    }

    meta = root / "app" / "src" / "main" / "java" / "com" / "example" / "flybrain" / "GeneratedConnectomeMeta.kt"
    motor_role_counts = {int(k): int(v) for k,v in selected.groupby("motor_role").size().to_dict().items()}

    meta.write_text('package com.example.flybrain\n\nobject GeneratedConnectomeMeta {\n    const val VERSION = "MaleCNS v1.0 · FlyBrain V1.06"\n    const val FORMAT_MAGIC = "FBC102"\n    const val FORMAT_VERSION = 102\n    const val NEURONS = %d\n    const val EDGES = %d\n    const val CONTACTS_RETAINED = %dL\n    const val VIS_START = %d\n    const val VIS_END = %d\n    const val OLF_START = %d\n    const val OLF_END = %d\n    const val GUST_START = %d\n    const val GUST_END = %d\n    const val MECH_START = %d\n    const val MECH_END = %d\n    const val DESC_START = %d\n    const val DESC_END = %d\n    const val ASC_START = %d\n    const val ASC_END = %d\n    const val VMOTOR_START = %d\n    const val VMOTOR_END = %d\n    const val OTHER_START = %d\n    const val OTHER_END = %d\n    const val MOTOR_LEG = 1\n    const val MOTOR_WING = 2\n    const val MOTOR_HALTERE = 3\n    const val MOTOR_NECK = 4\n    const val MOTOR_ABDOMEN = 5\n    const val MOTOR_JUMP = 6\n    const val MOTOR_OTHER = 7\n}\n' % (TARGET, len(edges), contacts,
       population_ranges["visual"][0], population_ranges["visual"][1], population_ranges["olfactory"][0], population_ranges["olfactory"][1],
       ranges["gustatory"][0], ranges["gustatory"][1], ranges["mechanosensory"][0], ranges["mechanosensory"][1],
       desc[0], desc[1], asc[0], asc[1], vmotor[0], vmotor[1], other[0], other[1]))

    selected_route_counts = {
        "forward_nonzero": int((route_forward_selected > 0).sum()),
        "turn_nonzero": int((route_turn_selected > 0).sum()),
        "escape_nonzero": int((route_escape_selected > 0).sum()),
        "escape_high": int((route_escape_selected >= 0.25).sum()),
    }

    source_route_counts = {
        "forward_nonzero": int((route_forward > 0).sum()),
        "turn_nonzero": int((route_turn > 0).sum()),
        "escape_nonzero": int((route_escape > 0).sum()),
    }

    report = {
        "dataset": "MaleCNS v1.0",
        "flybrain_version": "1.06",
        "binary_format": "FBC102",
        "node_record_bytes": NODE_SIZE,
        "edge_record_bytes": EDGE_SIZE,
        "selection": "exactly 16,669 traced annotated neurons; all descending and VNC motor neurons are retained; published neuron types are represented where possible; measured two-hop sensor-to-DN and DN-to-intermediate-to-motor route cells are preferentially retained by functional family; remaining quota is stratified by superclass and ranked by route score then degree",
        "source": BASE,
        "neurons_source": int(total),
        "neurons_retained": TARGET,
        "edges_retained": len(edges),
        "candidate_edges_between_retained_neurons": candidate_edges,
        "unresolved_edges_omitted": unresolved_edges,
        "contacts_retained_signed": contacts,
        "superclass_counts_source": {str(k): int(v) for k,v in counts.items()},
        "superclass_extra_quota": {str(k): int(v) for k,v in extra_quota.items()},
        "channel_ranges": ranges,
        "population_ranges": {k: [int(v[0]), int(v[1])] for k, v in population_ranges.items()},
        "edge_signs": NT_SIGN,
        "glutamate_modeling_convention": "glutamatergic edges are treated as inhibitory in the reduced dynamical model; receptor-specific exceptions are not represented",
        "unresolved_edges_policy": "edges with no recognized transmitter sign are omitted from direct current",
        "motor_role_counts": motor_role_counts,
        "descending_role_counts": {int(k): int(v) for k,v in selected.groupby("descending_role").size().to_dict().items()},
        "descending_role_source_counts": {
            str(i): int(role_counts_source[i]) for i in range(5)
        },
        "descending_type_role_counts": {
            f"{k[0]}|role{k[1]}": int(v)
            for k, v in selected[selected["superclass"].astype(str).eq("descending_neuron")]
            .groupby(["type", "descending_role"]).size().to_dict().items()
        },
        "motor_role_definition": "derived from curated annotation text for vnc_motor cells; runtime movement is driven only by measured vnc_motor activity",
        "descending_role_definition": "published behavioural cell-type names plus conservative annotation keywords; descriptive metadata only and never a synthetic current source",
        "route_score_definition": "route-family topology scores combine measured sensor->DN and DN->candidate->motor components from published edges; forward uses all sensory modalities, turn uses visual input, escape uses visual/mechanosensory threat input; scores are selection/readout metadata only and never create edges",
        "route_score_source_counts": source_route_counts,
        "route_score_selected_counts": selected_route_counts,
        "route_quota_requested": route_quota,
        "retained_sensor_to_desc_edges": int(retained_sensor_desc_edges),
        "retained_desc_to_motor_edges": int(retained_desc_motor_edges),
        "retained_sensor_to_desc_contacts": int(retained_sensor_desc.sum()),
        "retained_desc_to_motor_contacts": int(retained_desc_motor.sum()),
        "retained_desc_to_intermediate_edges": int(retained_desc_to_intermediate_edges),
        "retained_intermediate_to_motor_edges": int(retained_intermediate_to_motor_edges),
        "retained_desc_to_intermediate_to_motor_paths": int(retained_desc_to_intermediate_to_motor_paths),
        "retained_desc_to_intermediate_to_motor_by_role_paths": retained_two_hop_by_role.tolist(),
        "retained_sensor_to_desc_by_channel_role_contacts": retained_sensor_desc.tolist(),
        "retained_desc_to_motor_by_role_contacts": retained_desc_motor.tolist(),
        "direct_sensor_desc_full_graph_weight": direct_sensor_desc.tolist(),
        "direct_desc_motor_full_graph_weight": direct_desc_motor.tolist(),
        "license": "CC-BY-4.0",
        "download": "https://male-cns.janelia.org/download/",
        "paper": "Berg et al., Cell 2026, DOI 10.1016/j.cell.2026.08.015",
    }
    (root / "app" / "src" / "main" / "res" / "raw" / "malecns_reduced_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
