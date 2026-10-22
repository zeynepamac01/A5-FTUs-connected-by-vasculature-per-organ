"""Biology-first organ vasculature diagrams.

Every Table_S1 PathVessel row, including PathStep 0, remains a vessel node.
FTUs are local containment overlays around their internal PathStep-0 vessels.
They never replace a capillary or other vascular target. Vessel identity is
independent of whether it supplies or drains an FTU. Anatomical classifications
are checked against HRA ASCT+B v2.5.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd
import vl_convert as vlc


HRA_RELEASE_URL = (
    "https://cdn.humanatlas.io/hra-asctb-json-releases/"
    "hra-asctb-all.v2.5.json"
)
HEART_ID = "UBERON:0000948"
HEART_NODE_ID = f"heart:{HEART_ID}"

ORGAN_CONFIG = {
    "kidney": ("Kidney", "kidney_ftu_vascular_data.csv", "kidney"),
    "liver": ("Liver", "liver_ftu_vascular_data.csv", "liver"),
    "large_intestine": (
        "Large intestine",
        "l_intestine_ftu_vascular_data.csv",
        "large-intestine",
    ),
    "small_intestine": (
        "Small intestine",
        "s_intestine_ftu_vascular_data.csv",
        "small-intestine",
    ),
    "mouth": ("Mouth", "mouth_ftu_vascular_data.csv", "mouth"),
    "prostate": ("Prostate", "prostate_ftu_vascular_data.csv", "prostate"),
    "lung": ("Lung", "lung_ftu_vascular_data.csv", "lung"),
    "skin": ("Skin", "skin_ftu_vascular_data.csv", "skin"),
}
ORGAN_ORDER = list(ORGAN_CONFIG)

REQUIRED_COLUMNS = [
    "FTU",
    "Organ",
    "FTUID",
    "Vessel",
    "VesselID",
    "PathStep",
    "PathVessel",
    "PathVesselID",
    "Type",
]
TABLE_COLUMNS = [
    "FTU",
    "FTUID",
    "Vessel",
    "VesselID",
    "PathStep",
    "PathVessel",
    "PathVesselID",
]

INK = "#111111"
MUTED_INK = "#5c5c5c"
RED = "#d73027"
BLUE = "#2878b5"
CAPILLARY_PURPLE = "#7b3294"
PORTAL_PURPLE = "#984ea3"
NEUTRAL = "#6b6b6b"
FTU_FILL = "#f7f7f7"

TYPE_STYLE = {
    "artery": {"color": RED, "shape": "circle"},
    "arteriole": {"color": RED, "shape": "triangle-up"},
    "capillary": {"color": CAPILLARY_PURPLE, "shape": "diamond"},
    # Dr. Griffin's convention: capillaries are purple, while sinusoids are
    # grouped with venules and veins in blue.
    "sinusoid": {"color": BLUE, "shape": "diamond"},
    "venule": {"color": BLUE, "shape": "triangle-down"},
    "vein": {"color": BLUE, "shape": "circle"},
    "portal_vein": {"color": PORTAL_PURPLE, "shape": "cross"},
    "heart_chamber": {"color": NEUTRAL, "shape": "triangle-up"},
}


def _clean(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _fallback_id(name: str) -> str:
    slug = _slug(name)
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
    return f"missing:{slug}:{digest}"


def vessel_node_id(ontology_id: str, name: str) -> str:
    """Biological vessel identity never depends on route direction or color."""
    return f"vessel:{_clean(ontology_id) or _fallback_id(_clean(name))}"


def _own_name_classification(name: str) -> str | None:
    value = name.lower().strip()
    if "portal" in value and ("vein" in value or "venule" in value):
        return "portal_vein"
    if "sinusoid" in value:
        return "sinusoid"
    if "capillary" in value or "vasa recta" in value:
        return "capillary"
    if "venule" in value:
        return "venule"
    if "arteriole" in value:
        return "arteriole"
    if "venous plexus" in value:
        return "vein"
    if "artery" in value or "aorta" in value or "aortic" in value:
        return "artery"
    if "vein" in value or "vena cava" in value or value.startswith("vena "):
        return "vein"
    if any(
        chamber in value
        for chamber in ("left ventricle", "right ventricle", "left atrium", "right atrium")
    ):
        return "heart_chamber"
    if any(
        plexus in value
        for plexus in ("mucous plexus", "cutaneous plexus", "subpapillary plexus")
    ):
        return "capillary"
    return None


def build_hra_index(hra_release: dict) -> dict:
    """Index HRA anatomical paths and derive FTU containment evidence."""
    id_paths = defaultdict(list)
    name_paths = defaultdict(list)
    sheet_rows = {}
    for sheet_name, sheet in hra_release.items():
        rows = sheet.get("data", [])
        sheet_rows[sheet_name] = rows
        for row in rows:
            path = row.get("anatomical_structures", [])
            for structure in path:
                ontology_id = _clean(structure.get("id"))
                if ontology_id:
                    id_paths[ontology_id].append(path)
                for label in {
                    _clean(structure.get("name")),
                    _clean(structure.get("rdfs_label")),
                } - {""}:
                    name_paths[label.lower()].append(path)
    return {
        "id_paths": id_paths,
        "name_paths": name_paths,
        "sheet_rows": sheet_rows,
    }


def classify_vessel(
    name: str,
    ontology_id: str,
    is_target: bool,
    hra_index: dict,
) -> tuple[str, str]:
    """Return anatomical type and evidence; route direction is never the type."""
    direct = _own_name_classification(name)
    if direct:
        return direct, "vessel name"

    paths = (
        hra_index["id_paths"].get(ontology_id, [])
        if ontology_id
        else hra_index["name_paths"].get(name.lower(), [])
    )
    for path in paths:
        labels = " > ".join(
            _clean(item.get("name") or item.get("rdfs_label")).lower()
            for item in path
        )
        if "hepatic portal vein" in labels or "hepatic portal venule" in labels:
            return "portal_vein", "HRA portal-system path"

    for path in paths:
        matching_indices = [
            index
            for index, item in enumerate(path)
            if (
                ontology_id
                and _clean(item.get("id")) == ontology_id
            )
            or (
                not ontology_id
                and name.lower()
                in {
                    _clean(item.get("name")).lower(),
                    _clean(item.get("rdfs_label")).lower(),
                }
            )
        ]
        for index in matching_indices:
            for item in reversed(path[: index + 1]):
                inferred = _own_name_classification(
                    _clean(item.get("name") or item.get("rdfs_label"))
                )
                if inferred:
                    return inferred, "nearest typed HRA ancestor"

    if is_target and "plexus" in name.lower():
        return "capillary", "Table_S1 FTU-target plexus"
    raise ValueError(
        f"Cannot classify vessel without direction inference: {name!r} [{ontology_id}]"
    )


def _node_color(vessel_type: str, name: str, organ_key: str) -> str:
    """Color anatomy, with the conventional pulmonary oxygenation reversal."""
    lowered = name.lower()
    if organ_key == "lung":
        if "pulmonary" in lowered and vessel_type in {"artery", "arteriole"}:
            return BLUE
        if "pulmonary" in lowered and vessel_type in {"vein", "venule"}:
            return RED
        if "right ventricle" in lowered:
            return BLUE
        if "left atrium" in lowered:
            return RED
    return TYPE_STYLE[vessel_type]["color"]


def read_organ_csv(csv_dir: str | Path, organ_key: str) -> tuple[pd.DataFrame, Path]:
    csv_dir = Path(csv_dir)
    display, filename, _ = ORGAN_CONFIG[organ_key]
    source_path = csv_dir / filename
    frame = pd.read_csv(source_path, dtype=str).fillna("")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename}: missing required columns {missing}")
    for column in REQUIRED_COLUMNS:
        if column != "PathStep":
            frame[column] = frame[column].map(_clean)
    frame["PathStep_num"] = pd.to_numeric(frame["PathStep"], errors="raise").astype(int)
    return frame, source_path


def prepare_routes(frame: pd.DataFrame, organ_key: str) -> list[dict]:
    routes = []
    grouping = ["FTUID", "FTU", "Vessel", "VesselID"]
    for route_number, (key, group) in enumerate(
        frame.groupby(grouping, sort=True, dropna=False), 1
    ):
        ftu_id, ftu_name, target_name, target_id = map(_clean, key)
        ordered = group.sort_values("PathStep_num", kind="stable").copy()
        steps = ordered["PathStep_num"].tolist()
        if steps.count(0) != 1 or len(steps) != len(set(steps)):
            raise ValueError(f"{organ_key}/{ftu_id}/{target_name}: invalid PathStep sequence")
        expected = list(range(min(steps), max(steps) + 1))
        if steps != expected:
            raise ValueError(f"{organ_key}/{ftu_id}/{target_name}: PathStep gap")
        zero = ordered.loc[ordered["PathStep_num"] == 0].iloc[0]
        if _clean(zero["PathVessel"]) != target_name:
            raise ValueError(f"{organ_key}/{ftu_id}: step-zero target name mismatch")
        if target_id and _clean(zero["PathVesselID"]) != target_id:
            raise ValueError(f"{organ_key}/{ftu_id}: step-zero target ID mismatch")
        routes.append(
            {
                "route_id": f"{organ_key}:route:{route_number:02d}",
                "ftu_id": ftu_id,
                "ftu_name": ftu_name,
                "target_name": target_name,
                "target_id": target_id,
                "rows": ordered,
            }
        )
    return routes


def derive_ftu_hierarchy(
    organ_key: str,
    ftu_pairs: list[tuple[str, str]],
    hra_index: dict,
) -> dict[str, str | None]:
    """Use explicit HRA anatomical ancestry; never infer containment by name."""
    _, _, hra_sheet = ORGAN_CONFIG[organ_key]
    ftu_ids = {ftu_id for ftu_id, _ in ftu_pairs}
    evidence = defaultdict(list)
    for row in hra_index["sheet_rows"].get(hra_sheet, []):
        path_ids = [_clean(item.get("id")) for item in row.get("anatomical_structures", [])]
        for child_index, child_id in enumerate(path_ids):
            if child_id not in ftu_ids:
                continue
            ancestors = [item for item in path_ids[:child_index] if item in ftu_ids]
            if ancestors:
                evidence[child_id].append(ancestors[-1])
    hierarchy = {}
    for ftu_id in ftu_ids:
        candidates = evidence.get(ftu_id, [])
        hierarchy[ftu_id] = max(set(candidates), key=candidates.count) if candidates else None
    return hierarchy


def _curve_geometry(source: dict, target: dict, arrow_t: float = 0.70) -> dict:
    midpoint = (source["x"] + target["x"]) / 2
    path = (
        f"M{source['x']:.2f},{source['y']:.2f} "
        f"C{midpoint:.2f},{source['y']:.2f} "
        f"{midpoint:.2f},{target['y']:.2f} "
        f"{target['x']:.2f},{target['y']:.2f}"
    )
    p0 = (source["x"], source["y"])
    p1 = (midpoint, source["y"])
    p2 = (midpoint, target["y"])
    p3 = (target["x"], target["y"])

    # Split the Bezier curve at its midpoint. Drawing the source and target
    # halves separately makes a capillary a genuinely purple vessel segment,
    # rather than merely a purple label or point.
    def lerp(a, b, ratio=0.5):
        return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)

    q0, q1, q2 = lerp(p0, p1), lerp(p1, p2), lerp(p2, p3)
    r0, r1 = lerp(q0, q1), lerp(q1, q2)
    split = lerp(r0, r1)
    source_half_path = (
        f"M{p0[0]:.2f},{p0[1]:.2f} C{q0[0]:.2f},{q0[1]:.2f} "
        f"{r0[0]:.2f},{r0[1]:.2f} {split[0]:.2f},{split[1]:.2f}"
    )
    target_half_path = (
        f"M{split[0]:.2f},{split[1]:.2f} C{r1[0]:.2f},{r1[1]:.2f} "
        f"{q2[0]:.2f},{q2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}"
    )
    t = arrow_t
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    dx = (
        3 * u**2 * (p1[0] - p0[0])
        + 6 * u * t * (p2[0] - p1[0])
        + 3 * t**2 * (p3[0] - p2[0])
    )
    dy = (
        3 * u**2 * (p1[1] - p0[1])
        + 6 * u * t * (p2[1] - p1[1])
        + 3 * t**2 * (p3[1] - p2[1])
    )
    return {
        "path": path,
        "source_half_path": source_half_path,
        "target_half_path": target_half_path,
        "arrow_x": round(x, 2),
        "arrow_y": round(y, 2),
        "arrow_angle": round(math.degrees(math.atan2(dy, dx)) + 90, 2),
    }


def _ftu_overlays(
    organ_key: str,
    ftu_pairs: list[tuple[str, str]],
    hierarchy: dict[str, str | None],
    target_memberships: dict[str, set[str]],
    node_lookup: dict[str, dict],
) -> list[dict]:
    """Build local FTU containment boundaries around PathStep-0 vessels."""
    names = dict(ftu_pairs)
    members_by_ftu = defaultdict(set)
    for vessel_id, ftu_ids in target_memberships.items():
        for ftu_id in ftu_ids:
            members_by_ftu[ftu_id].add(vessel_id)

    children = defaultdict(list)
    for child_id, parent_id in hierarchy.items():
        if parent_id:
            children[parent_id].append(child_id)

    def descendants(ftu_id: str) -> set[str]:
        result = set()
        for child_id in children[ftu_id]:
            result.add(child_id)
            result.update(descendants(child_id))
        return result

    def depth(ftu_id: str) -> int:
        value = 0
        parent_id = hierarchy.get(ftu_id)
        while parent_id:
            value += 1
            parent_id = hierarchy.get(parent_id)
        return value

    overlays = []
    for ftu_id, name in ftu_pairs:
        direct_members = sorted(members_by_ftu[ftu_id])
        geometry_members = set(direct_members)
        contained_ftus = descendants(ftu_id)
        for child_id in contained_ftus:
            geometry_members.update(members_by_ftu[child_id])
        member_nodes = [node_lookup[node_id] for node_id in sorted(geometry_members)]
        child_count = len(children[ftu_id])
        nested_depth = depth(ftu_id)
        pad_x = 52.0 if child_count else 28.0
        pad_y = 48.0 if child_count else 27.0
        x = min(node["x"] for node in member_nodes) - pad_x
        x2 = max(node["x"] for node in member_nodes) + pad_x
        y = min(node["y"] for node in member_nodes) - pad_y
        y2 = max(node["y"] for node in member_nodes) + pad_y
        overlays.append(
            {
                "id": f"ftu:{ftu_id}",
                "ontology_id": ftu_id,
                "name": name,
                "parent_id": hierarchy.get(ftu_id) or "",
                "depth": nested_depth,
                "contains_children": bool(child_count),
                "show_container": bool(child_count),
                "member_vessel_ids": direct_members,
                "geometry_vessel_ids": sorted(geometry_members),
                "x": round(x, 2),
                "x2": round(x2, 2),
                "y": round(y, 2),
                "y2": round(y2, 2),
                "marker_x": round(x2 + 24, 2),
                "marker_y": round((y + y2) / 2, 2),
                "label_x": round(x + 10 if child_count else x2 + 42, 2),
                "label_y": round(y + 18 if child_count else (y + y2) / 2, 2),
                "label_baseline": "middle",
                "label_font_size": 18 if child_count else 16,
                "stroke_width": 1.7 if child_count else 1.15,
                "stroke_dash": [],
                "stroke_opacity": 0.82,
                "fill_opacity": 0.0,
            }
        )
    if organ_key == "kidney":
        # The nephron is the sole enclosing boundary. Its constituent FTUs and
        # the collecting-duct FTUs are local text labels, not separate boxes or
        # vessel-connected black squares.
        kidney_label_positions = {
            "renal corpuscle": (2050.0, 320.0),
            "descending limb of loop of henle": (2050.0, 650.0),
            "loop of henle ascending limb thin segment": (2050.0, 540.0),
            "thick ascending limb of loop of henle": (2050.0, 780.0),
            "cortical collecting duct": (2290.0, 710.0),
            "inner medullary collecting duct": (2290.0, 430.0),
            "outer medullary collecting duct": (2290.0, 570.0),
        }
        for overlay in overlays:
            lowered_name = overlay["name"].lower()
            if lowered_name == "nephron":
                overlay["x2"] = 2200.0
                overlay["y"] = min(overlay["y"], 285.0)
                overlay["y2"] = max(overlay["y2"], 835.0)
                overlay["label_x"] = overlay["x"] + 12.0
                overlay["label_y"] = overlay["y"] + 21.0
            if lowered_name != "nephron":
                overlay["stroke_opacity"] = 0.0
                overlay["fill_opacity"] = 0.0
                overlay["marker_x"], overlay["marker_y"] = kidney_label_positions[lowered_name]
                overlay["label_x"] = overlay["marker_x"] + 18
                overlay["label_y"] = overlay["marker_y"]
                overlay["label_font_size"] = 14
            if lowered_name == "renal corpuscle":
                overlay["show_container"] = True
                overlay["x"] = 1450.0
                overlay["x2"] = 1850.0
                overlay["y"] = 300.0
                overlay["y2"] = 395.0
                overlay["label_x"] = 1462.0
                overlay["label_y"] = 380.0
                overlay["label_font_size"] = 15
                overlay["stroke_opacity"] = 0.82
                overlay["fill_opacity"] = 0.0
                overlay["stroke_width"] = 1.4
    elif organ_key == "liver":
        # The lobule contains several internal vessels. A containment boundary
        # communicates that structure without a spoke of pseudo-flow lines.
        for overlay in overlays:
            if overlay["name"].lower() == "liver lobule":
                overlay["show_container"] = True
                overlay["x"] = 1550.0
                overlay["x2"] = 2020.0
                overlay["y"] = 350.0
                overlay["y2"] = 665.0
                overlay["label_x"] = 1565.0
                overlay["label_y"] = 375.0
                overlay["label_font_size"] = 18
                overlay["stroke_width"] = 1.7
                overlay["fill_opacity"] = 0.0
    elif organ_key == "skin":
        for overlay in overlays:
            lowered_name = overlay["name"].lower()
            if lowered_name == "hair follicle":
                overlay["x"] = 1580.0
                overlay["x2"] = 1935.0
                overlay["y"] = 535.0
                overlay["y2"] = 690.0
                overlay["label_x"] = 1595.0
                overlay["label_y"] = 560.0
            if lowered_name == "dermal papilla":
                overlay["show_container"] = True
                overlay["x"], overlay["x2"] = 1625.0, 1735.0
                overlay["y"], overlay["y2"] = 565.0, 655.0
                overlay["label_x"], overlay["label_y"] = 1755.0, 610.0
                overlay["stroke_opacity"] = 0.82
                overlay["fill_opacity"] = 0.0
                overlay["stroke_width"] = 1.4
                overlay["label_font_size"] = 14
            elif lowered_name == "epidermal ridge of digit":
                overlay["show_container"] = True
                overlay["x"], overlay["x2"] = 1625.0, 1735.0
                overlay["y"], overlay["y2"] = 395.0, 490.0
                overlay["label_x"], overlay["label_y"] = 1755.0, 442.5
                overlay["stroke_opacity"] = 0.82
                overlay["fill_opacity"] = 0.0
                overlay["stroke_width"] = 1.4
                overlay["label_font_size"] = 14
    else:
        # Reference-inspired hollow FTU boxes: the authoritative PathStep-0
        # vessel is physically inside its FTU boundary. The box is annotation,
        # never a substitute flow node or an invented edge.
        for overlay in overlays:
            member_nodes = [node_lookup[node_id] for node_id in overlay["member_vessel_ids"]]
            overlay["show_container"] = True
            overlay["x"] = round(min(node["x"] for node in member_nodes) - 55.0, 2)
            overlay["x2"] = round(max(node["x"] for node in member_nodes) + 55.0, 2)
            overlay["y"] = round(min(node["y"] for node in member_nodes) - 46.0, 2)
            overlay["y2"] = round(max(node["y"] for node in member_nodes) + 46.0, 2)
            overlay["label_x"] = overlay["x2"] + 18.0
            overlay["label_y"] = round((overlay["y"] + overlay["y2"]) / 2, 2)
            overlay["label_baseline"] = "middle"
            overlay["label_font_size"] = 16
            overlay["stroke_width"] = 1.5
            overlay["stroke_opacity"] = 0.86
            overlay["fill_opacity"] = 0.0
    # Draw enclosing parents first, then their nested children.
    return sorted(overlays, key=lambda item: (item["depth"], item["name"]))


def build_organ_model(
    csv_dir: str | Path,
    organ_key: str,
    hra_release: dict,
) -> dict:
    frame, source_path = read_organ_csv(csv_dir, organ_key)
    routes = prepare_routes(frame, organ_key)
    hra_index = build_hra_index(hra_release)
    route_count = len(routes)
    # Preserve the publication layout requirement: the Vega drawing area is
    # exactly 3000 px wide (10 inches at 300 px/in).
    width = 3000
    height = max(1050, 500 + route_count * (34 if organ_key == "kidney" else 44))
    center_y = height / 2
    track_gap = 210
    lane_gap = 22 if organ_key == "kidney" else 46
    route_span = (route_count - 1) * lane_gap
    artery_lanes = [center_y - track_gap - route_span / 2 + i * lane_gap for i in range(route_count)]
    vein_lanes = [center_y + track_gap - route_span / 2 + i * lane_gap for i in range(route_count)]

    ftu_pairs_frame = frame[["FTUID", "FTU"]].drop_duplicates().sort_values(["FTU", "FTUID"])
    ftu_pairs = [(row.FTUID, row.FTU) for row in ftu_pairs_frame.itertuples(index=False)]
    hierarchy = derive_ftu_hierarchy(organ_key, ftu_pairs, hra_index)

    node_meta = {}
    occurrences = defaultdict(list)
    target_memberships = defaultdict(set)
    route_records = []
    classification_records = []

    for route_index, route in enumerate(routes):
        rows = route["rows"]
        negative = rows.loc[rows["PathStep_num"] < 0]
        positive = rows.loc[rows["PathStep_num"] > 0]
        ordered_ids = []
        for row in rows.to_dict("records"):
            step = int(row["PathStep_num"])
            name = _clean(row["PathVessel"])
            ontology_id = _clean(row["PathVesselID"])
            node_id = vessel_node_id(ontology_id, name)
            vessel_type, evidence = classify_vessel(
                name, ontology_id, step == 0, hra_index
            )
            if step < 0:
                fraction = (step - int(negative["PathStep_num"].min())) / max(1, len(negative) - 1)
                x = 230 + fraction * 1180
                y = artery_lanes[route_index]
                flow_position = "supplies"
            elif step > 0:
                fraction = (step - 1) / max(1, len(positive) - 1)
                x = 1410 - fraction * 1180
                y = vein_lanes[route_index]
                flow_position = "drains"
            else:
                x = 1960
                y = center_y - 110 + route_index * min(36, 220 / max(1, route_count - 1))
                flow_position = "inside_ftu"
                target_memberships[node_id].add(route["ftu_id"])

            proposed = {
                "id": node_id,
                "name": name,
                "ontology_id": ontology_id,
                "type": "vessel",
                "vessel_type": vessel_type,
                "classification_evidence": evidence,
                "color": _node_color(vessel_type, name, organ_key),
                # A single vessel-node symbol keeps the diagram readable. The
                # exact anatomical type remains encoded by color and label.
                "shape": "circle",
            }
            if node_id in node_meta:
                existing = node_meta[node_id]
                if (existing["name"], existing["ontology_id"]) != (name, ontology_id):
                    raise ValueError(f"{organ_key}: conflicting identity for {node_id}")
                if existing["vessel_type"] != vessel_type:
                    raise ValueError(
                        f"{organ_key}: conflicting vessel types for {name}: "
                        f"{existing['vessel_type']} versus {vessel_type}"
                    )
            else:
                node_meta[node_id] = proposed
            occurrences[node_id].append(
                {
                    "x": x,
                    "y": y,
                    "route_id": route["route_id"],
                    "step": step,
                    "flow_position": flow_position,
                }
            )
            classification_records.append(
                {
                    "organ": organ_key,
                    "route_id": route["route_id"],
                    "PathStep": step,
                    "name": name,
                    "ontology_id": ontology_id,
                    "vessel_type": vessel_type,
                    "evidence": evidence,
                    "is_ftu_target": step == 0,
                }
            )
            ordered_ids.append(node_id)
        route_records.append({**route, "ordered_node_ids": ordered_ids})

    nodes = [
        {
            "id": HEART_NODE_ID,
            "name": "Heart",
            "ontology_id": HEART_ID,
            "type": "heart",
            "color": INK,
            "shape": "triangle-up",
            "x": 110.0,
            "y": center_y,
        }
    ]
    occupied = set()
    for node_id, values in occurrences.items():
        meta = dict(node_meta[node_id])
        target_values = [item for item in values if item["step"] == 0]
        position_values = target_values or values
        x = statistics.median(item["x"] for item in position_values)
        y = statistics.mean(item["y"] for item in position_values)
        while (round(x, 1), round(y, 1)) in occupied:
            y += 10
        occupied.add((round(x, 1), round(y, 1)))
        meta.update(
            {
                "x": round(x, 2),
                "y": round(y, 2),
                "route_count": len({item["route_id"] for item in values}),
                "occurrence_count": len(values),
                "is_ftu_target": bool(target_values),
                "ftu_ids": sorted(target_memberships.get(node_id, set())),
                "label": meta["name"],
                "label_x": round(x, 2),
                "label_y": round(y - 12 if y <= center_y else y + 12, 2),
                "label_baseline": "bottom" if y <= center_y else "top",
                "label_angle": -42,
                "label_align": "center",
            }
        )
        nodes.append(meta)

    # Give distinct internal vessels enough local room for FTU containment
    # boundaries and labels. This changes layout only, never route topology.
    target_nodes = sorted(
        (node for node in nodes if node.get("is_ftu_target")),
        key=lambda node: (node["y"], node["name"]),
    )
    if target_nodes:
        target_gap = 72.0 if organ_key == "kidney" else 120.0
        target_span = target_gap * (len(target_nodes) - 1)
        start_y = center_y - target_span / 2
        for index, node in enumerate(target_nodes):
            node["y"] = round(start_y + index * target_gap, 2)
            node["label_x"] = node["x"]
            node["label_y"] = round(
                node["y"] - 17,
                2,
            )
            node["label_baseline"] = "bottom"
            node["label_angle"] = 0
            node["label_align"] = "center"

    if organ_key == "kidney":
        kidney_target_layout = {
            "renal afferent arteriole": (1500.0, center_y - 205.0, center_y - 219.0),
            "glomerular capillary": (1650.0, center_y - 205.0, center_y - 236.0),
            "renal efferent arteriole": (1800.0, center_y - 205.0, center_y - 219.0),
            "descending vasa recta of kidney": (1900.0, center_y - 85.0, center_y - 102.0),
            "vasa recta of kidney": (2020.0, center_y, center_y - 17.0),
            "ascending vasa recta of kidney": (1900.0, center_y + 95.0, center_y + 78.0),
            "peritubular capillary": (1650.0, center_y + 215.0, center_y + 198.0),
        }
        for node in target_nodes:
            position = kidney_target_layout.get(node["name"].lower())
            if not position:
                continue
            node["x"], node["y"], node["label_y"] = position
            node["label_x"] = node["x"]
            node["label_baseline"] = "bottom"
            node["label_angle"] = 0
            node["label_align"] = "center"
    elif organ_key == "liver":
        liver_target_layout = {
            "hepatic arteriole": (1620.0, 430.0, 412.0, "bottom"),
            "hepatic portal venule": (1620.0, 590.0, 608.0, "top"),
            "hepatic sinusoid": (1780.0, 510.0, 492.0, "bottom"),
            "central vein of liver": (1940.0, 510.0, 528.0, "top"),
        }
        for node in target_nodes:
            position = liver_target_layout.get(node["name"].lower())
            if not position:
                continue
            node["x"], node["y"], node["label_y"], node["label_baseline"] = position
            node["label_x"] = node["x"]
            node["label_angle"] = 0
            node["label_align"] = "center"
    elif organ_key == "skin":
        skin_target_layout = {
            "branch of subpapillary plexus": (1680.0, 442.5),
            "branch of deep cutaneous plexus": (1680.0, 610.0),
        }
        for node in target_nodes:
            position = skin_target_layout.get(node["name"].lower())
            if not position:
                continue
            node["x"], node["y"] = position
            node["label_x"] = node["x"]
            node["label_y"] = node["y"] - 16.0
            node["label_baseline"] = "bottom"
            node["label_angle"] = 0
            node["label_align"] = "center"

    # Alternate label tiers along each vascular track. This preserves the node
    # coordinates and ordering while preventing neighboring long vessel names
    # from being printed on top of one another.
    for upper_side in (True, False):
        side_nodes = sorted(
            (
                node for node in nodes
                if node["type"] == "vessel"
                and not node.get("is_ftu_target")
                and ((node["y"] <= center_y) == upper_side)
            ),
            key=lambda node: (node["x"], node["y"], node["name"]),
        )
        tier_count = 3 if len(side_nodes) >= 14 else 2
        for index, node in enumerate(side_nodes):
            tier = index % tier_count
            offset = 13.0 + tier * 34.0
            node["label_x"] = node["x"]
            node["label_y"] = round(
                node["y"] - offset if upper_side else node["y"] + offset,
                2,
            )
            node["label_baseline"] = "bottom" if upper_side else "top"
            node["label_angle"] = -42
            node["label_align"] = "center"

    node_lookup = {node["id"]: node for node in nodes}
    ftu_overlays = _ftu_overlays(
        organ_key,
        ftu_pairs,
        hierarchy,
        target_memberships,
        node_lookup,
    )
    ftu_containers = [overlay for overlay in ftu_overlays if overlay["show_container"]]
    ftu_markers = [overlay for overlay in ftu_overlays if not overlay["show_container"]]
    ftu_marker_links = []
    for overlay in ftu_markers:
        for vessel_id in overlay["member_vessel_ids"]:
            vessel = node_lookup[vessel_id]
            ftu_marker_links.append(
                {
                    "vessel_id": vessel_id,
                    "ftu_id": overlay["id"],
                    "path": (
                        f"M{vessel['x']:.2f},{vessel['y']:.2f} "
                        f"L{overlay['marker_x']:.2f},{overlay['marker_y']:.2f}"
                    ),
                }
            )
    edge_routes = defaultdict(set)
    edge_roles = defaultdict(set)
    for route in route_records:
        path = [HEART_NODE_ID, *route["ordered_node_ids"], HEART_NODE_ID]
        target_index = 1 + next(
            index
            for index, row in enumerate(route["rows"].to_dict("records"))
            if int(row["PathStep_num"]) == 0
        )
        for index, (source_id, target_id) in enumerate(zip(path, path[1:])):
            edge = (source_id, target_id)
            edge_routes[edge].add(route["route_id"])
            edge_roles[edge].add("supplies" if index < target_index else "drains")

    links = []
    for edge_number, ((source_id, target_id), route_ids) in enumerate(edge_routes.items(), 1):
        source = node_lookup[source_id]
        target = node_lookup[target_id]
        flow_roles = sorted(edge_roles[(source_id, target_id)])
        link = {
            "id": f"{organ_key}:edge:{edge_number:03d}",
            "source_id": source_id,
            "target_id": target_id,
            "routes": sorted(route_ids),
            "route_count": len(route_ids),
            "flow_roles": flow_roles,
            "source_color": source["color"],
            "target_color": target["color"],
            "arrow_color": target["color"],
        }
        link.update(_curve_geometry(source, target))
        links.append(link)

    model = {
        "organ_key": organ_key,
        "display_name": ORGAN_CONFIG[organ_key][0],
        "source_path": source_path,
        "width": width,
        "height": height,
        "nodes": nodes,
        "links": links,
        "ftu_overlays": ftu_overlays,
        "ftu_containers": ftu_containers,
        "ftu_markers": ftu_markers,
        "ftu_marker_links": ftu_marker_links,
        "ftu_hierarchy": hierarchy,
        "routes": route_records,
        "classification_records": classification_records,
        "target_memberships": {
            node_id: sorted(ftu_ids) for node_id, ftu_ids in target_memberships.items()
        },
        "validation": {},
    }
    model["validation"] = validate_model(frame, model)
    return model


def validate_model(frame: pd.DataFrame, model: dict) -> dict:
    source_nodes = {
        vessel_node_id(row.PathVesselID, row.PathVessel)
        for row in frame.itertuples(index=False)
    } | {HEART_NODE_ID}
    visual_nodes = {node["id"] for node in model["nodes"]}
    source_edges = set()
    for route in model["routes"]:
        path = [HEART_NODE_ID, *route["ordered_node_ids"], HEART_NODE_ID]
        source_edges.update(zip(path, path[1:]))
    visual_edges = {(edge["source_id"], edge["target_id"]) for edge in model["links"]}
    target_rows = frame.loc[frame["PathStep_num"] == 0]
    expected_targets = {
        vessel_node_id(row.PathVesselID, row.PathVessel)
        for row in target_rows.itertuples(index=False)
    }
    displayed_targets = {
        node["id"] for node in model["nodes"] if node.get("is_ftu_target")
    }
    expected_ftu_memberships = {
        (vessel_id, f"ftu:{ftu_id}")
        for vessel_id, ftu_ids in model["target_memberships"].items()
        for ftu_id in ftu_ids
    }
    visual_ftu_memberships = {
        (vessel_id, overlay["id"])
        for overlay in model["ftu_overlays"]
        for vessel_id in overlay["member_vessel_ids"]
    }
    expected_ftu_ids = {f"ftu:{ftu_id}" for ftu_ids in model["target_memberships"].values() for ftu_id in ftu_ids}
    displayed_ftu_ids = {overlay["id"] for overlay in model["ftu_overlays"]}
    overlay_ids = [overlay["id"] for overlay in model["ftu_overlays"]]
    expected_marker_memberships = {
        (vessel_id, overlay["id"])
        for overlay in model["ftu_markers"]
        for vessel_id in overlay["member_vessel_ids"]
    }
    visual_marker_memberships = {
        (link["vessel_id"], link["ftu_id"])
        for link in model["ftu_marker_links"]
    }
    flow_ftu_edges = [
        edge for edge in visual_edges if edge[0].startswith("ftu:") or edge[1].startswith("ftu:")
    ]
    duplicate_coordinates = defaultdict(list)
    for node in model["nodes"]:
        duplicate_coordinates[(node["x"], node["y"])].append(node["id"])
    duplicate_coordinates = {
        key: value for key, value in duplicate_coordinates.items() if len(value) > 1
    }
    hepatic_sinusoid_nodes = [
        node for node in model["nodes"] if node["name"].lower() == "hepatic sinusoid"
    ]
    result = {
        "missing_source_nodes": sorted(source_nodes - visual_nodes),
        "extra_visualization_nodes": sorted(visual_nodes - source_nodes),
        "missing_source_edges": sorted(source_edges - visual_edges),
        "extra_visualization_edges": sorted(visual_edges - source_edges),
        "missing_target_vessels": sorted(expected_targets - displayed_targets),
        "extra_target_vessels": sorted(displayed_targets - expected_targets),
        "missing_ftu_memberships": sorted(expected_ftu_memberships - visual_ftu_memberships),
        "extra_ftu_memberships": sorted(visual_ftu_memberships - expected_ftu_memberships),
        "missing_ftu_overlays": sorted(expected_ftu_ids - displayed_ftu_ids),
        "extra_ftu_overlays": sorted(displayed_ftu_ids - expected_ftu_ids),
        "duplicate_ftu_overlays": len(overlay_ids) - len(set(overlay_ids)),
        "missing_ftu_marker_links": sorted(expected_marker_memberships - visual_marker_memberships),
        "extra_ftu_marker_links": sorted(visual_marker_memberships - expected_marker_memberships),
        "flow_edges_to_ftu_containers": flow_ftu_edges,
        "duplicate_coordinates": duplicate_coordinates,
        "unclassified_vessels": [
            node["id"]
            for node in model["nodes"]
            if node["type"] == "vessel" and node["vessel_type"] not in TYPE_STYLE
        ],
        "hepatic_sinusoid_node_count": len(hepatic_sinusoid_nodes),
        "passed": False,
    }
    errors = []
    for key in (
        "missing_source_nodes",
        "extra_visualization_nodes",
        "missing_source_edges",
        "extra_visualization_edges",
        "missing_target_vessels",
        "extra_target_vessels",
        "missing_ftu_memberships",
        "extra_ftu_memberships",
        "missing_ftu_overlays",
        "extra_ftu_overlays",
        "duplicate_ftu_overlays",
        "missing_ftu_marker_links",
        "extra_ftu_marker_links",
        "flow_edges_to_ftu_containers",
        "duplicate_coordinates",
        "unclassified_vessels",
    ):
        if result[key]:
            count = len(result[key]) if hasattr(result[key], "__len__") else result[key]
            errors.append(f"{key}: {count}")
    if model["organ_key"] == "liver" and len(hepatic_sinusoid_nodes) != 1:
        errors.append(
            f"hepatic_sinusoid_node_count: {len(hepatic_sinusoid_nodes)} (expected 1)"
        )
    result["errors"] = errors
    result["passed"] = not errors
    return result


def make_vega_spec(model: dict) -> dict:
    nodes = model["nodes"]
    vessels = [node for node in nodes if node["type"] == "vessel"]
    heart = [node for node in nodes if node["type"] == "heart"]
    width = model["width"]
    height = model["height"]
    link_halves = []
    for link in model["links"]:
        width_value = 1.1 + min(link["route_count"], 10) * 0.22
        def half_record(path, color):
            is_purple = color in {CAPILLARY_PURPLE, PORTAL_PURPLE}
            return {
                "path": path,
                "color": color,
                "stroke_width": round(width_value * (0.86 if is_purple else 1.0), 3),
                "stroke_opacity": 0.56 if is_purple else 0.76,
            }
        link_halves.extend(
            [
                half_record(link["source_half_path"], link["source_color"]),
                half_record(link["target_half_path"], link["target_color"]),
            ]
        )
        link["arrow_opacity"] = (
            0.56
            if link["arrow_color"] in {CAPILLARY_PURPLE, PORTAL_PURPLE}
            else 0.72
        )
    if model["organ_key"] == "lung":
        legend_entries = [
            ("Pulmonary arterial vessel", BLUE, "circle"),
            ("Capillary", CAPILLARY_PURPLE, "circle"),
            ("Sinusoid", BLUE, "circle"),
            ("Pulmonary venous vessel", RED, "circle"),
            ("Portal vein", PORTAL_PURPLE, "circle"),
        ]
    else:
        legend_entries = [
            ("Arterial vessel", RED, "circle"),
            ("Capillary", CAPILLARY_PURPLE, "circle"),
            ("Sinusoid", BLUE, "circle"),
            ("Venous vessel", BLUE, "circle"),
            ("Portal vein", PORTAL_PURPLE, "circle"),
        ]
    legend = [
        {
            "label": label,
            "color": color,
            "shape": shape,
            "x": width - 365,
            "y": 42 + index * 29,
        }
        for index, (label, color, shape) in enumerate(legend_entries)
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": (
            f"{model['display_name']}: biology-first Table_S1 vascular routes with "
            "explicit FTU target vessels and HRA-v2.5 FTU containment"
        ),
        "width": width,
        "height": height,
        "padding": 0,
        "autosize": "none",
        "background": "#ffffff",
        "usermeta": {
            "source_csv": model["source_path"].name,
            "hra_release": HRA_RELEASE_URL,
            "validation": model["validation"],
            "biology_rule": (
                "PathStep 0 remains a vessel; FTUs are local containment overlays; "
                "vessel type is independent of supply/drain direction"
            ),
        },
        "data": [
            {"name": "links", "values": model["links"]},
            {"name": "link_halves", "values": link_halves},
            {"name": "vessels", "values": vessels},
            {"name": "heart", "values": heart},
            {"name": "ftu_containers", "values": model["ftu_containers"]},
            {"name": "ftu_markers", "values": model["ftu_markers"]},
            {"name": "ftu_marker_links", "values": model["ftu_marker_links"]},
            {"name": "legend", "values": legend},
        ],
        "marks": [
            {
                "type": "text",
                "encode": {"enter": {
                    "x": {"value": 36}, "y": {"value": 42},
                    "text": {"value": model["display_name"]},
                    "font": {"value": "Arial"}, "fontSize": {"value": 30},
                    "fontWeight": {"value": 600}, "fill": {"value": INK},
                }},
            },
            {
                "type": "text",
                "encode": {"enter": {
                    "x": {"value": 36}, "y": {"value": 70},
                    "text": {"value": "Every source vessel retained; FTUs contain their internal vessels"},
                    "font": {"value": "Arial"}, "fontSize": {"value": 15},
                    "fill": {"value": MUTED_INK},
                }},
            },
            {
                "type": "rect", "from": {"data": "ftu_containers"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "x2": {"field": "x2"},
                    "y": {"field": "y"}, "y2": {"field": "y2"},
                    "fill": {"value": FTU_FILL}, "fillOpacity": {"field": "fill_opacity"},
                    "stroke": {"value": MUTED_INK}, "strokeWidth": {"field": "stroke_width"},
                    "strokeDash": {"field": "stroke_dash"}, "strokeOpacity": {"field": "stroke_opacity"},
                }},
            },
            {
                "type": "path", "from": {"data": "ftu_marker_links"},
                "encode": {"enter": {
                    "path": {"field": "path"}, "fill": {"value": None},
                    "stroke": {"value": MUTED_INK}, "strokeWidth": {"value": 1.0},
                    "strokeOpacity": {"value": 0.38},
                }},
            },
            {
                "type": "path", "from": {"data": "link_halves"},
                "encode": {"enter": {
                    "path": {"field": "path"}, "fill": {"value": None},
                    "stroke": {"field": "color"}, "strokeWidth": {"field": "stroke_width"},
                    "strokeOpacity": {"field": "stroke_opacity"},
                }},
            },
            {
                "type": "symbol", "from": {"data": "links"},
                "encode": {"enter": {
                    "x": {"field": "arrow_x"}, "y": {"field": "arrow_y"},
                    "shape": {"value": "triangle-up"}, "angle": {"field": "arrow_angle"},
                    "size": {"value": 34}, "fill": {"field": "arrow_color"},
                    "fillOpacity": {"field": "arrow_opacity"},
                }},
            },
            {
                "type": "symbol", "from": {"data": "vessels"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "y": {"field": "y"},
                    "shape": {"field": "shape"}, "size": {"value": 88},
                    "fill": {"field": "color"}, "stroke": {"value": "#ffffff"},
                    "strokeWidth": {"value": 0.8},
                }},
            },
            {
                "type": "text", "from": {"data": "vessels"},
                "encode": {"enter": {
                    "x": {"field": "label_x"}, "y": {"field": "label_y"},
                    "text": {"field": "label"}, "angle": {"field": "label_angle"},
                    "align": {"field": "label_align"}, "baseline": {"field": "label_baseline"},
                    "font": {"value": "Arial"}, "fontSize": {"value": 15},
                    "fill": {"value": INK}, "limit": {"value": 360},
                }},
            },
            {
                "type": "text", "from": {"data": "ftu_containers"},
                "encode": {"enter": {
                    "x": {"field": "label_x"}, "y": {"field": "label_y"},
                    "text": {"field": "name"}, "baseline": {"field": "label_baseline"},
                    "font": {"value": "Arial"}, "fontSize": {"field": "label_font_size"},
                    "fontWeight": {"value": 600}, "fill": {"value": INK},
                    "limit": {"value": 690},
                }},
            },
            {
                "type": "symbol", "from": {"data": "ftu_markers"},
                "encode": {"enter": {
                    "x": {"field": "marker_x"}, "y": {"field": "marker_y"},
                    "shape": {"value": "square"}, "size": {"value": 150},
                    "fill": {"value": "#ffffff"}, "stroke": {"value": INK},
                    "strokeWidth": {"value": 1.8},
                }},
            },
            {
                "type": "text", "from": {"data": "ftu_markers"},
                "encode": {"enter": {
                    "x": {"field": "label_x"}, "y": {"field": "label_y"},
                    "text": {"field": "name"}, "baseline": {"value": "middle"},
                    "font": {"value": "Arial"}, "fontSize": {"field": "label_font_size"},
                    "fontWeight": {"value": 600}, "fill": {"value": INK},
                    "limit": {"value": 690},
                }},
            },
            {
                "type": "symbol", "from": {"data": "heart"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "y": {"field": "y"},
                    "shape": {"field": "shape"}, "size": {"value": 230},
                    "fill": {"field": "color"},
                }},
            },
            {
                "type": "text", "from": {"data": "heart"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "y": {"field": "y"},
                    "dx": {"value": 14}, "dy": {"value": -12},
                    "text": {"field": "name"}, "font": {"value": "Arial"},
                    "fontSize": {"value": 16}, "fontWeight": {"value": 600},
                    "fill": {"value": INK},
                }},
            },
            {
                "type": "text",
                "encode": {"enter": {
                    "x": {"value": width - 385}, "y": {"value": 18},
                    "text": {"value": "Vessel type"}, "font": {"value": "Arial"},
                    "fontSize": {"value": 16}, "fontWeight": {"value": 600},
                    "fill": {"value": INK},
                }},
            },
            {
                "type": "symbol", "from": {"data": "legend"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "y": {"field": "y"},
                    "shape": {"field": "shape"}, "size": {"value": 90},
                    "fill": {"field": "color"},
                }},
            },
            {
                "type": "text", "from": {"data": "legend"},
                "encode": {"enter": {
                    "x": {"field": "x"}, "dx": {"value": 18}, "y": {"field": "y"},
                    "text": {"field": "label"}, "baseline": {"value": "middle"},
                    "font": {"value": "Arial"}, "fontSize": {"value": 15},
                    "fill": {"value": INK},
                }},
            },
        ],
        "config": {},
    }


def make_illustrator_friendly_svg(svg_text: str) -> str:
    """Create explicit, movable Illustrator groups without flattening paths.

    A biological route necessarily changes SVG objects where its stroke color
    changes. Keeping those path objects intact makes individual curves easy to
    edit, while the named groups let Illustrator move either the flow or the
    complete diagram without disconnecting its constituent objects.
    """
    if 'id="editable-diagram"' in svg_text:
        return svg_text

    group_pattern = re.compile(
        r'(<g class="mark-path role-mark"[^>]*>)(.*?)(</g>)',
        re.DOTALL,
    )
    path_pattern = re.compile(r'<path\b[^>]*/>')
    flow_match = None
    for match in group_pattern.finditer(svg_text):
        body = match.group(2)
        if path_pattern.findall(body) and any(
            color in body for color in (RED, BLUE, CAPILLARY_PURPLE, PORTAL_PURPLE)
        ):
            flow_match = match
            break
    if flow_match is None:
        raise ValueError("Could not identify the vascular flow path group in SVG")

    mark_pattern = re.compile(
        r'<g class="mark-(?:path|symbol|text|rect) role-mark"[^>]*>.*?</g>',
        re.DOTALL,
    )
    marks = list(mark_pattern.finditer(svg_text))
    flow_index = next(
        (index for index, mark in enumerate(marks) if mark.start() == flow_match.start()),
        None,
    )
    if flow_index is None or flow_index < 2 or flow_index + 8 >= len(marks):
        raise ValueError("Unexpected Vega mark order while creating Illustrator groups")

    # Stable mark order from make_vega_spec:
    # FTU containers, FTU annotation links, flow, arrows, vessel nodes/labels,
    # FTU labels/markers, then the heart node/label.
    diagram_start = marks[flow_index - 2].start()
    arrow_match = marks[flow_index + 1]
    diagram_end = marks[flow_index + 8].end()
    flow_markup = flow_match.group(0)

    return (
        svg_text[:diagram_start]
        + '<g id="editable-diagram" data-illustrator-group="complete editable diagram">'
        + svg_text[diagram_start : flow_match.start()]
        + '<g id="vascular-flow" data-illustrator-group="flow paths and arrowheads">'
        + flow_markup
        + svg_text[flow_match.end() : arrow_match.start()]
        + arrow_match.group(0)
        + "</g>"
        + svg_text[arrow_match.end() : diagram_end]
        + "</g>"
        + svg_text[diagram_end:]
    )


def render_all(
    csv_dir: str | Path,
    json_dir: str | Path,
    svg_dir: str | Path,
    hra_release: dict,
) -> dict:
    json_dir = Path(json_dir)
    svg_dir = Path(svg_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    models = {
        organ_key: build_organ_model(csv_dir, organ_key, hra_release)
        for organ_key in ORGAN_ORDER
    }
    failures = {
        key: model["validation"]["errors"]
        for key, model in models.items()
        if not model["validation"]["passed"]
    }
    if failures:
        raise RuntimeError(f"Biological validation blocked rendering: {failures}")
    for organ_key, model in models.items():
        spec = make_vega_spec(model)
        stem = f"{organ_key}_vasculature_ftus_biological"
        json_path = json_dir / f"{stem}.json"
        svg_path = svg_dir / f"{stem}.svg"
        json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
        svg_text = make_illustrator_friendly_svg(vlc.vega_to_svg(vg_spec=spec))
        svg_path.write_text(svg_text)
        results[organ_key] = {
            "model": model,
            "spec": spec,
            "json": json_path,
            "svg": svg_path,
        }
    return results
