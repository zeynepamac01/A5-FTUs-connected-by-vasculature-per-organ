from __future__ import annotations

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import vl_convert as vlc

import organ_vasculature_biological_vega as base
import organ_vasculature_reference_style as previous


OUTPUT_STEM = "vasculature_ftus_reference_style_scientific"
CANVAS_WIDTH = 3000
CANVAS_HEIGHT = 1242
CANVAS_WIDTH_IN = 10
CANVAS_HEIGHT_IN = 4.14
SHOW_LEGEND = False  # Retained for the future introductory legend page.

# The SVG viewBox is 3000 units wide and is displayed at 720 points.  These
# values therefore render as the requested Illustrator measurements.
PT_SCALE = CANVAS_WIDTH / (CANVAS_WIDTH_IN * 72)
LINE_WIDTH = 1.0 * PT_SCALE
LABEL_SIZE = 7 * PT_SCALE
NODE_SIZE = 300
OPEN_NODE_STROKE = 1.0 * PT_SCALE
FTU_CORNER_RADIUS = 24


def _single_line(value: str) -> str:
    """Keep labels easy to select and edit as one Illustrator text line."""
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def _rounded_polyline_geometry(
    source: dict,
    target: dict,
    elbow_x: float | None = None,
    radius: float = 36.0,
) -> dict:
    """Build an orthogonal connector with genuine rounded elbow curves."""
    sx, sy = float(source["x"]), float(source["y"])
    tx, ty = float(target["x"]), float(target["y"])
    if abs(sy - ty) < 1e-6:
        points = [(sx, sy), (tx, ty)]
    else:
        middle_x = (sx + tx) / 2 if elbow_x is None else float(elbow_x)
        points = [(sx, sy), (middle_x, sy), (middle_x, ty), (tx, ty)]
    points = [
        point for index, point in enumerate(points)
        if index == 0 or point != points[index - 1]
    ]

    lengths = [math.dist(left, right) for left, right in zip(points, points[1:])]
    total = sum(lengths)

    def interpolate(distance: float):
        remaining = distance
        for left, right in zip(points, points[1:]):
            length = math.dist(left, right)
            if length == 0:
                continue
            if remaining <= length:
                ratio = remaining / length
                point = (
                    left[0] + (right[0] - left[0]) * ratio,
                    left[1] + (right[1] - left[1]) * ratio,
                )
                return point, left, right
            remaining -= length
        return points[-1], points[-2], points[-1]

    split, _, _ = interpolate(total / 2)
    arrow, arrow_left, arrow_right = interpolate(total * 0.63)

    source_points = [points[0]]
    travelled = 0.0
    for index, length in enumerate(lengths):
        if travelled + length < total / 2 - 1e-6:
            source_points.append(points[index + 1])
            travelled += length
        else:
            break
    if source_points[-1] != split:
        source_points.append(split)

    target_points = [split]
    travelled = 0.0
    for index, length in enumerate(lengths):
        next_travelled = travelled + length
        if next_travelled > total / 2 + 1e-6:
            target_points.append(points[index + 1])
        travelled = next_travelled
    if target_points[-1] != points[-1]:
        target_points.append(points[-1])

    def rounded_path(values: list[tuple[float, float]]) -> str:
        if len(values) < 2:
            return ""
        commands = [f"M{values[0][0]:.2f},{values[0][1]:.2f}"]
        for index in range(1, len(values) - 1):
            previous_point, corner, next_point = (
                values[index - 1], values[index], values[index + 1]
            )
            incoming = math.dist(previous_point, corner)
            outgoing = math.dist(corner, next_point)
            if incoming < 1e-6 or outgoing < 1e-6:
                commands.append(f"L{corner[0]:.2f},{corner[1]:.2f}")
                continue
            local_radius = min(radius, incoming / 2, outgoing / 2)
            before = (
                corner[0] + (previous_point[0] - corner[0]) * local_radius / incoming,
                corner[1] + (previous_point[1] - corner[1]) * local_radius / incoming,
            )
            after = (
                corner[0] + (next_point[0] - corner[0]) * local_radius / outgoing,
                corner[1] + (next_point[1] - corner[1]) * local_radius / outgoing,
            )
            incoming_unit = (
                (corner[0] - previous_point[0]) / incoming,
                (corner[1] - previous_point[1]) / incoming,
            )
            outgoing_unit = (
                (next_point[0] - corner[0]) / outgoing,
                (next_point[1] - corner[1]) / outgoing,
            )
            # Cubic approximation of a true quarter-circle.  This gives each
            # elbow consistent curvature and tangent continuity in Illustrator.
            handle = local_radius * 0.5522847498
            control_one = (
                before[0] + incoming_unit[0] * handle,
                before[1] + incoming_unit[1] * handle,
            )
            control_two = (
                after[0] - outgoing_unit[0] * handle,
                after[1] - outgoing_unit[1] * handle,
            )
            commands.append(f"L{before[0]:.2f},{before[1]:.2f}")
            commands.append(
                f"C{control_one[0]:.2f},{control_one[1]:.2f} "
                f"{control_two[0]:.2f},{control_two[1]:.2f} "
                f"{after[0]:.2f},{after[1]:.2f}"
            )
        commands.append(f"L{values[-1][0]:.2f},{values[-1][1]:.2f}")
        return " ".join(commands)

    dx = arrow_right[0] - arrow_left[0]
    dy = arrow_right[1] - arrow_left[1]
    return {
        "path": rounded_path(points),
        "source_half_path": rounded_path(source_points),
        "target_half_path": rounded_path(target_points),
        "arrow_x": round(arrow[0], 2),
        "arrow_y": round(arrow[1], 2),
        "arrow_angle": round(math.degrees(math.atan2(dy, dx)) + 90, 2),
    }


def _lookup(model: dict) -> dict[str, dict]:
    return {
        node["name"].lower(): node
        for node in model["nodes"]
        if node["type"] == "vessel"
    }


def _overlay_lookup(model: dict) -> dict[str, dict]:
    return {item["name"].lower(): item for item in model["ftu_overlays"]}


def _set_node(node: dict, x: float, y: float, *, label_dx=0, label_dy=-54,
              baseline="bottom", angle=0, align="center"):
    node["x"], node["y"] = float(x), float(y)
    node["label_x"] = float(x + label_dx)
    node["label_y"] = float(y + label_dy)
    node["label_baseline"] = baseline
    node["label_angle"] = angle
    node["label_align"] = align
    node["label"] = _single_line(node["name"])


def _set_box(item: dict, x: float, x2: float, y: float, y2: float,
             label_x: float | None = None, label_y: float | None = None):
    item["display_label"] = _single_line(item["name"])
    item.update(
        {
            "show_container": True,
            "x": float(x),
            "x2": float(x2),
            "y": float(y),
            "y2": float(y2),
            "label_x": float(label_x if label_x is not None else x + 22),
            "label_y": float(label_y if label_y is not None else y + 38),
            "label_baseline": "middle",
            "label_font_size": LABEL_SIZE,
            "fill_opacity": 0.0,
            "stroke_width": 5.0,
            "stroke_opacity": 0.72,
            "corner_radius": FTU_CORNER_RADIUS,
        }
    )


def _set_marker(item: dict, x: float, y: float):
    item["display_label"] = _single_line(item["name"])
    item.update(
        {
            "show_container": False,
            "marker_x": float(x),
            "marker_y": float(y),
            "label_x": float(x + 45),
            "label_y": float(y),
            "label_baseline": "middle",
            "label_font_size": LABEL_SIZE,
        }
    )


def _reachable(start_ids: set[str], links: list[dict], reverse=False) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for link in links:
        left, right = link["source_id"], link["target_id"]
        if reverse:
            left, right = right, left
        adjacency.setdefault(left, set()).add(right)
    reached = set(start_ids)
    frontier = list(start_ids)
    while frontier:
        current = frontier.pop()
        for target in adjacency.get(current, set()):
            if target not in reached:
                reached.add(target)
                frontier.append(target)
    return reached


def _reachable_until(
    start_ids: set[str], links: list[dict], stop_ids: set[str], blocked_ids: set[str]
) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for link in links:
        adjacency.setdefault(link["source_id"], set()).add(link["target_id"])
    reached = set(start_ids)
    frontier = list(start_ids)
    while frontier:
        current = frontier.pop()
        if current in stop_ids:
            continue
        for target in adjacency.get(current, set()):
            if target in blocked_ids or target in reached:
                continue
            reached.add(target)
            frontier.append(target)
    return reached


def _wrap_label(value: str, width: int = 17) -> str:
    words = value.split()
    if len(value) <= width or len(words) == 1:
        return value
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


COMMON_VISIBLE = {
    "left ventricle",
    "right atrium",
    "aorta",
}

ORGAN_VISIBLE = {
    "prostate": {
        "common iliac artery", "prostatic artery", "prostatic venous plexus",
    },
    "small_intestine": {
        "superior mesenteric artery", "jejunal arcade",
    },
    "large_intestine": {
        "inferior mesenteric artery",
        "submucosal artery of colon",
    },
    "liver": {
        "splenic vein",
    },
    "lung": {
        "right ventricle", "left atrium", "pulmonary trunk", "pulmonary vein",
        "left ventricle", "right atrium",
    },
    "mouth": {
        "external carotid artery",
    },
    "skin": {
        "subclavian artery", "subclavian vein",
    },
    "kidney": {
        "renal artery", "renal vein",
    },
}


def _apply_scientific_classes(model: dict):
    """Apply Griffin's node semantics without altering source topology."""
    nodes = [n for n in model["nodes"] if n["type"] == "vessel"]
    node_by_id = {n["id"]: n for n in nodes}

    # Intestinal/colonic vasa recta are arterial; kidney vasa recta are capillaries.
    if model["organ_key"] in {"small_intestine", "large_intestine"}:
        for node in nodes:
            if "vasa recta" in node["name"].lower():
                node["vessel_type"] = "artery"
                node["classification_evidence"] = "Dr. Griffin organ-specific correction"

    # Every non-capillary vessel between a first capillary bed and the hepatic
    # sinusoid is a portal-system vessel, even if its anatomical name says vein.
    sinusoid_ids = {
        n["id"] for n in nodes if n["name"].lower() == "hepatic sinusoid"
    }
    if sinusoid_ids:
        capillary_ids = {
            n["id"] for n in nodes
            if n["vessel_type"] == "capillary" and n["id"] not in sinusoid_ids
        }
        heart_ids = {n["id"] for n in model["nodes"] if n["type"] == "heart"}
        downstream = _reachable_until(
            capillary_ids, model["links"], sinusoid_ids, heart_ids
        )
        portal_ids = downstream & set(node_by_id)
        for node_id in portal_ids:
            node = node_by_id[node_id]
            if node["vessel_type"] not in {"capillary", "sinusoid"}:
                node["vessel_type"] = "portal_vein"
                node["classification_evidence"] = "portal route after first capillary bed"

    for node in nodes:
        vessel_type = node["vessel_type"]
        name = node["name"].lower()
        node["shape"] = "circle"
        node["fill"] = base.TYPE_STYLE[vessel_type]["color"]
        node["stroke"] = "#ffffff"
        node["node_stroke_width"] = 2.0
        node["color"] = base.TYPE_STYLE[vessel_type]["color"]

        if vessel_type == "sinusoid":
            node.update(
                fill="#ffffff",
                stroke=base.BLUE,
                node_stroke_width=OPEN_NODE_STROKE,
                color=base.BLUE,
            )
        elif vessel_type == "portal_vein":
            node.update(
                fill="#ffffff",
                stroke=base.PORTAL_PURPLE,
                node_stroke_width=OPEN_NODE_STROKE,
                color=base.PORTAL_PURPLE,
            )
        elif vessel_type == "capillary":
            node.update(fill=base.CAPILLARY_PURPLE, color=base.CAPILLARY_PURPLE)
        elif vessel_type in {"vein", "venule"}:
            node.update(fill=base.BLUE, color=base.BLUE)
        elif vessel_type in {"artery", "arteriole"}:
            node.update(fill=base.RED, color=base.RED)
        elif vessel_type == "heart_chamber":
            node.update(fill=base.NEUTRAL, color=base.NEUTRAL)

        if model["organ_key"] == "lung" and "pulmonary" in name:
            if vessel_type in {"artery", "arteriole"}:
                node.update(shape="square", fill=base.BLUE, stroke="#ffffff", color=base.BLUE)
            elif vessel_type in {"vein", "venule"}:
                node.update(shape="square", fill=base.RED, stroke="#ffffff", color=base.RED)


def _compress_main_tracks(model: dict):
    """Use roughly half the old non-FTU spacing and nested label tiers."""
    vessels = [n for n in model["nodes"] if n["type"] == "vessel"]
    heart = next(n for n in model["nodes"] if n["type"] == "heart")
    heart["x"], heart["y"] = 175.0, CANVAS_HEIGHT / 2

    ordinary = [n for n in vessels if not n.get("is_ftu_target")]
    xs = [float(n["x"]) for n in ordinary]
    low, high = min(xs), max(xs)
    for node in ordinary:
        ratio = 0 if high == low else (float(node["x"]) - low) / (high - low)
        node["x"] = 340 + ratio * 1120
        upper = float(node["y"]) < 475
        # Compact arterial and return tracks with only a small lane offset.
        node["y"] = (
            620 + (float(node["y"]) - 300) * 0.30
            if upper
            else 800 + (float(node["y"]) - 650) * 0.30
        )

    dense = model["organ_key"] in {"liver", "mouth", "lung", "skin"}
    angle = -67 if dense else -48
    tiers = (42, 92, 142, 192) if dense else (42, 92, 142)
    for upper in (True, False):
        row = sorted(
            (
                n for n in ordinary
                if n.get("label") and (float(n["y"]) < CANVAS_HEIGHT / 2) == upper
            ),
            key=lambda n: (n["x"], n["name"]),
        )
        for index, node in enumerate(row):
            # Keep every label visibly tied to its node.  A shallow two-level
            # stagger is enough to distinguish neighbors without creating the
            # detached-label effect of the earlier multi-tier layout.
            magnitude = tiers[index % len(tiers)]
            signed_offset = -magnitude if upper else magnitude
            node["label"] = _single_line(node["name"])
            node["label_x"] = node["x"]
            node["label_y"] = node["y"] + signed_offset
            node["label_baseline"] = "bottom" if signed_offset < 0 else "top"
            node["label_angle"] = angle
            node["label_align"] = "center"


def _layout_prostate(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_node(n["prostatic arteriole"], 2050, 620, label_dy=-62)
    _set_node(n["prostatic capillary"], 2050, 700, label_dx=45, label_dy=38,
              baseline="top", align="left")
    _set_node(n["prostatic venule"], 2050, 800, label_dy=58, baseline="top")
    _set_box(f["prostate glandular acinus"], 1720, 2400, 660, 760)


def _layout_small_intestine(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_node(n["vasa recta of jejunum"], 1800, 620, label_dy=-62)
    _set_node(n["mucous plexus of small intestine"], 1800, 700,
              label_dx=-50, label_dy=0, baseline="middle", align="right")
    _set_node(n["branch of mucous plexus of small intestine"], 2200, 700,
              label_dy=48, baseline="top")
    _set_node(n["jejunal vein"], 1800, 800, label_dy=58, baseline="top")
    _set_box(f["intestinal villus"], 2020, 2400, 650, 760)


def _layout_large_intestine(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_node(n["mucosal arteriole of colon"], 2050, 620, label_dy=-62)
    _set_node(n["branch of mucous plexus of colon"], 2050, 700,
              label_dx=45, label_dy=38, baseline="top", align="left")
    _set_node(n["mucosal venule of colon"], 2050, 800, label_dy=58, baseline="top")
    _set_box(f["crypt of lieberkuhn"], 1700, 2400, 650, 760)


def _layout_liver(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_node(n["hepatic arteriole"], 1850, 650, label_dx=-38, label_dy=-58,
              align="right")
    _set_node(n["hepatic portal venule"], 2250, 650, label_dy=-58)
    _set_node(n["hepatic sinusoid"], 2050, 780, label_dx=50, label_dy=0,
              baseline="middle", align="left")
    _set_node(n["central vein of liver"], 2050, 920, label_dy=58, baseline="top")
    _set_box(f["liver lobule"], 1650, 2400, 500, 1020)


def _layout_lung(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    # Four visually separate circulation lanes make each heart connection
    # unambiguous while retaining every intermediate vessel node.
    for node in (x for x in model["nodes"] if x["type"] == "vessel" and not x.get("is_ftu_target")):
        name = node["name"].lower()
        if name in {"right ventricle", "pulmonary trunk"} or (
            "pulmonary" in name and node["vessel_type"] in {"artery", "arteriole"}
        ):
            node["y"] = 420.0
        elif name in {"left atrium", "pulmonary vein"} or (
            "pulmonary" in name and node["vessel_type"] in {"vein", "venule"}
        ):
            node["y"] = 620.0
        elif node["vessel_type"] in {"artery", "arteriole"} or name == "left ventricle":
            node["y"] = 790.0
        elif node["vessel_type"] in {"vein", "venule"} or name == "right atrium":
            node["y"] = 1030.0

    # Re-anchor every label after moving the vessels into their four lanes.
    # This preserves every source label without the old-coordinate collisions.
    lane_offsets = {
        420.0: (-42, -92, -142),
        620.0: (42, 92, 142),
        790.0: (-42, -92, -142),
        1030.0: (42, 92, 142),
    }
    for lane_y, offsets in lane_offsets.items():
        row = sorted(
            (
                node for node in model["nodes"]
                if node["type"] == "vessel"
                and not node.get("is_ftu_target")
                and float(node["y"]) == lane_y
            ),
            key=lambda node: (node["x"], node["name"]),
        )
        for index, node in enumerate(row):
            offset = offsets[index % len(offsets)]
            node["label"] = _single_line(node["name"])
            node["label_x"] = node["x"]
            node["label_y"] = lane_y + offset
            node["label_angle"] = -58
            node["label_align"] = "center"
            node["label_baseline"] = "bottom" if offset < 0 else "top"

    chamber_labels = {
        "right ventricle": (45, -38, "left"),
        "left atrium": (45, -38, "left"),
        "left ventricle": (45, -38, "left"),
        "right atrium": (45, 38, "left"),
    }
    for name, (dx, dy, align) in chamber_labels.items():
        node = n[name]
        node["label"] = _single_line(node["name"])
        node["label_x"], node["label_y"] = node["x"] + dx, node["y"] + dy
        node["label_angle"] = 0
        node["label_align"] = align
        node["label_baseline"] = "bottom" if dy < 0 else "top"

    # Griffin requested explicit systemic labels for the bronchial-gland loop
    # so it cannot be confused with the special pulmonary circulation.
    n["bronchial arteriole"]["label"] = "systemic arterial vessel"
    n["bronchial venule"]["label"] = "systemic venous vessel"

    _set_node(n["pre-alveolar capillary"], 1770, 315, label_dx=-48, label_dy=0,
              baseline="middle", align="right")
    _set_node(n["alveolar capillary"], 2200, 405, label_dx=0, label_dy=48,
              baseline="top", align="center")
    _set_node(n["post-alveolar capillary"], 1770, 525, label_dx=-48, label_dy=0,
              baseline="middle", align="right")
    _set_box(f["alveolus of lung"], 2030, 2400, 250, 500)

    _set_node(n["bronchial arteriole"], 1770, 760, label_dx=-48, label_dy=0,
              baseline="middle", align="right")
    _set_node(n["bronchial capillary"], 2200, 850, label_dx=0, label_dy=48,
              baseline="top", align="center")
    _set_node(n["bronchial venule"], 1770, 1030, label_dx=-48, label_dy=0,
              baseline="middle", align="right")
    _set_box(f["bronchus submucosal gland"], 2030, 2400, 700, 970)


def _layout_mouth(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    rows = [
        ("superior parotid gland", 345, "parotid ducto-acinar unit"),
        ("sublingual gland", 650, "sublingual ducto-acinar unit"),
        ("submandibular gland", 955, "submandibular ducto-acinar unit"),
    ]
    ftu_names = {
        "superior parotid gland": "parotid gland ducto-acinar unit",
        "sublingual gland": "sublingual gland ducto-acinar unit",
        "submandibular gland": "submandibular gland ducto-acinar unit",
    }
    for prefix, y, title in rows:
        _set_node(n[f"{prefix} arteriole"], 1740, y, label_dy=-52)
        n[f"{prefix} arteriole"]["label"] = "arteriole"
        _set_node(n[f"{prefix} capillary"], 2050, y, label_dy=48,
                  baseline="top", align="center")
        n[f"{prefix} capillary"]["label"] = "capillary"
        _set_node(n[f"{prefix} venule"], 2360, y, label_dy=-52)
        n[f"{prefix} venule"]["label"] = "venule"
        box = f[ftu_names[prefix]]
        _set_box(box, 1900, 2200, y - 105, y + 120, 1915, y - 70)
        box["display_label"] = _single_line(title)


def _layout_skin(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_node(n["cutaneous arteriole"], 1700, 300, label_dy=-58)
    _set_node(n["subpapillary plexus"], 2150, 650, label_dx=-30, label_dy=-48,
              baseline="bottom", align="right")
    # Dr. Griffin's sketch places one shared subpapillary branch inside the
    # overlapping dermal-papilla and epidermal-ridge FTU boundaries.
    _set_node(n["branch of subpapillary plexus"], 2450, 650,
              label_dy=42, baseline="top")
    _set_node(n["deep cutaneous plexus"], 1700, 650, label_dx=-30, label_dy=30,
              baseline="top", align="right")
    _set_node(n["branch of deep cutaneous plexus"], 1950, 650,
              label_dy=62, baseline="top")
    _set_node(n["cutaneous venule"], 1700, 1040, label_dy=58, baseline="top")
    _set_box(f["hair follicle"], 1760, 2070, 560, 740)
    # Offset and overlap these boundaries around their shared branch, matching
    # Dr. Griffin's sketch without implying two disconnected FTUs.
    epidermal = f["epidermal ridge of digit"]
    _set_box(epidermal, 2260, 2560, 590, 750, 2300, 790)
    epidermal["display_label"] = "epidermal ridge of digit"
    dermal = f["dermal papilla"]
    _set_box(dermal, 2220, 2520, 540, 700, 2240, 565)
    dermal["display_label"] = "dermal papilla"


def _layout_kidney(model: dict):
    n, f = _lookup(model), _overlay_lookup(model)
    _set_box(f["nephron"], 1580, 2480, 155, 1190)
    _set_box(f["renal corpuscle"], 1670, 2410, 235, 410)
    f["nephron"]["label_x"], f["nephron"]["label_y"] = 1600, 125
    f["renal corpuscle"]["label_x"], f["renal corpuscle"]["label_y"] = 1690, 205
    _set_node(n["renal afferent arteriole"], 1760, 345, label_dx=70,
              label_dy=-35, baseline="bottom", align="left")
    n["renal afferent arteriole"]["label"] = "afferent"
    _set_node(n["glomerular capillary"], 2040, 345, label_dy=56, baseline="top")
    _set_node(n["renal efferent arteriole"], 2320, 345, label_dx=-70,
              label_dy=-35, baseline="bottom", align="right")
    n["renal efferent arteriole"]["label"] = "efferent"

    _set_node(n["descending vasa recta of kidney"], 1770, 500, label_dx=-48,
              label_dy=0, baseline="middle", align="right")
    n["descending vasa recta of kidney"]["label"] = "descending vasa recta"
    _set_node(n["vasa recta of kidney"], 2050, 560, label_dx=45,
              label_dy=0, baseline="middle", align="left")
    n["vasa recta of kidney"]["label"] = "vasa recta"
    _set_node(n["ascending vasa recta of kidney"], 1770, 625, label_dx=-48,
              label_dy=0, baseline="middle", align="right")
    n["ascending vasa recta of kidney"]["label"] = "ascending vasa recta"
    _set_node(n["peritubular capillary"], 2260, 625, label_dy=52,
              baseline="top", align="center")

    marker_rows = {
        "descending limb of loop of henle": (760, "descending Henle limb"),
        "loop of henle ascending limb thin segment": (840, "thin ascending Henle limb"),
        "thick ascending limb of loop of henle": (920, "thick ascending Henle limb"),
        "cortical collecting duct": (1000, "cortical collecting duct"),
        "inner medullary collecting duct": (1080, "inner medullary collecting duct"),
        "outer medullary collecting duct": (1160, "outer medullary collecting duct"),
    }
    for name, (row, display_label) in marker_rows.items():
        _set_marker(f[name], 1640, row)
        f[name]["display_label"] = display_label


ORGAN_LAYOUTS = {
    "prostate": _layout_prostate,
    "small_intestine": _layout_small_intestine,
    "large_intestine": _layout_large_intestine,
    "liver": _layout_liver,
    "lung": _layout_lung,
    "mouth": _layout_mouth,
    "skin": _layout_skin,
    "kidney": _layout_kidney,
}


def _use_legend_space(model: dict):
    """Expand the FTU region into the space reserved for the hidden legend."""
    if SHOW_LEGEND:
        return
    origin = 1550.0
    factor = 1.35

    def expand(value: float) -> float:
        return origin + (float(value) - origin) * factor

    for node in model["nodes"]:
        if node["type"] == "vessel" and float(node["x"]) >= origin:
            node["x"] = expand(node["x"])
            node["label_x"] = expand(node["label_x"])
    for item in model["ftu_overlays"]:
        for field in ("x", "x2", "label_x", "marker_x"):
            if field in item and float(item[field]) >= origin:
                item[field] = expand(item[field])


def _rebuild_ftu_data(model: dict):
    model["ftu_containers"] = [o for o in model["ftu_overlays"] if o["show_container"]]
    model["ftu_markers"] = [o for o in model["ftu_overlays"] if not o["show_container"]]
    nodes = {n["id"]: n for n in model["nodes"]}
    model["ftu_marker_links"] = []
    for overlay in model["ftu_markers"]:
        for vessel_id in overlay["member_vessel_ids"]:
            vessel = nodes[vessel_id]
            model["ftu_marker_links"].append(
                {
                    "vessel_id": vessel_id,
                    "ftu_id": overlay["id"],
                    "path": (
                        f"M{vessel['x']:.2f},{vessel['y']:.2f} "
                        f"L{overlay['marker_x']:.2f},{overlay['marker_y']:.2f}"
                    ),
                    "stroke_opacity": 0.26,
                }
            )


def _simplify_kidney_marker_links(model: dict):
    if model["organ_key"] != "kidney":
        return
    nodes = _lookup(model)
    markers = {item["name"].replace("\n", " ").lower(): item for item in model["ftu_markers"]}
    representative = {
        "descending limb of loop of henle": "descending vasa recta of kidney",
        "loop of henle ascending limb thin segment": "vasa recta of kidney",
        "thick ascending limb of loop of henle": "peritubular capillary",
        "cortical collecting duct": "peritubular capillary",
        "inner medullary collecting duct": "descending vasa recta of kidney",
        "outer medullary collecting duct": "ascending vasa recta of kidney",
    }
    for record in model["ftu_marker_links"]:
        record["path"] = ""
        record["stroke_opacity"] = 0.0
    by_pair = {
        (record["vessel_id"], record["ftu_id"]): record
        for record in model["ftu_marker_links"]
    }
    for ftu_name, vessel_name in representative.items():
        marker = markers[ftu_name]
        vessel = nodes[vessel_name]
        record = by_pair.get((vessel["id"], marker["id"]))
        if record is None:
            continue
        elbow_x = marker["marker_x"] - 38
        record["path"] = (
            f"M{vessel['x']:.2f},{vessel['y']:.2f} "
            f"L{elbow_x:.2f},{vessel['y']:.2f} "
            f"L{elbow_x:.2f},{marker['marker_y']:.2f} "
            f"L{marker['marker_x']:.2f},{marker['marker_y']:.2f}"
        )
        record["stroke_opacity"] = 0.26


def revision_layout(validated: dict) -> dict:
    model = previous.reference_layout(validated)
    model["width"], model["height"] = CANVAS_WIDTH, CANVAS_HEIGHT
    _apply_scientific_classes(model)
    _compress_main_tracks(model)
    ORGAN_LAYOUTS[model["organ_key"]](model)
    _use_legend_space(model)
    _rebuild_ftu_data(model)
    _simplify_kidney_marker_links(model)

    node_by_id = {n["id"]: n for n in model["nodes"]}
    heart = next(n for n in model["nodes"] if n["type"] == "heart")
    for link in model["links"]:
        source, target = node_by_id[link["source_id"]], node_by_id[link["target_id"]]
        link["source_color"] = source.get("color", base.NEUTRAL)
        link["target_color"] = target.get("color", base.NEUTRAL)
        link["arrow_color"] = target.get("color", base.NEUTRAL)
        elbow_x = None
        if source["type"] == "heart" or target["type"] == "heart":
            # The chamber lines terminate at the heart rather than forming a
            # chamber-to-chamber trunk with the heart as a side branch.
            elbow_x = heart["x"]
        elif float(source["x"]) < 1550 <= float(target["x"]):
            # Enter the FTU region at the destination x-position.  This uses
            # one meaningful bend instead of a midpoint stair-step.
            elbow_x = float(target["x"])
        elif float(target["x"]) < 1550 <= float(source["x"]):
            # Leave the FTU region at the source x-position for the same
            # single-elbow treatment on return paths.
            elbow_x = float(source["x"])
        elif model["organ_key"] == "skin":
            names = (source["name"].lower(), target["name"].lower())
            if names == ("cutaneous arteriole", "subpapillary plexus"):
                elbow_x = float(target["x"])
            elif names == ("subpapillary plexus", "cutaneous venule"):
                elbow_x = float(source["x"])
        geometry = _rounded_polyline_geometry(source, target, elbow_x=elbow_x)
        link.update(geometry)

    # Exact source graph must remain unchanged.
    assert {n["id"] for n in model["nodes"]} == {n["id"] for n in validated["nodes"]}
    assert {(e["source_id"], e["target_id"]) for e in model["links"]} == {
        (e["source_id"], e["target_id"]) for e in validated["links"]
    }
    return model


def _data(spec: dict, name: str) -> list[dict]:
    return next(item["values"] for item in spec["data"] if item["name"] == name)


def make_revision_spec(model: dict) -> dict:
    spec = previous.make_reference_spec(model)
    spec["width"], spec["height"] = CANVAS_WIDTH, CANVAS_HEIGHT
    spec["description"] += "; Griffin/Katy revision"
    spec["usermeta"]["layout_variant"] = (
        "Griffin/Katy revision; 10 x 4.14 in; source topology unchanged"
    )

    marks = spec["marks"]
    marks[0]["encode"]["enter"].update(
        {"x": {"value": 70}, "y": {"value": 90}, "fontSize": {"value": 82}}
    )
    marks[1]["encode"]["enter"].update(
        {"x": {"value": 70}, "y": {"value": 140}, "fontSize": {"value": 42}}
    )
    marks[2]["encode"]["enter"].update(
        {
            "strokeWidth": {"value": 5},
            "cornerRadius": {"field": "corner_radius"},
        }
    )
    marks[3]["encode"]["enter"].update(
        {
            "strokeWidth": {"value": 3.5},
            "strokeOpacity": {"field": "stroke_opacity"},
            "strokeCap": {"value": "round"},
            "strokeJoin": {"value": "round"},
        }
    )
    marks[4]["encode"]["enter"].update(
        {
            "strokeWidth": {"value": LINE_WIDTH},
            "strokeOpacity": {"value": 1.0},
            "strokeCap": {"value": "round"},
            "strokeJoin": {"value": "round"},
        }
    )
    marks[5]["encode"]["enter"].update(
        {"size": {"value": 1}, "fillOpacity": {"value": 0}}
    )
    marks[6]["encode"]["enter"].update(
        {
            "shape": {"field": "shape"},
            "size": {"value": NODE_SIZE},
            "fill": {"field": "fill"},
            "stroke": {"field": "stroke"},
            "strokeWidth": {"field": "node_stroke_width"},
        }
    )
    marks[7]["encode"]["enter"].update(
        {"fontSize": {"value": LABEL_SIZE}, "fill": {"field": "color"},
         "limit": {"value": 1200}}
    )
    marks[8]["encode"]["enter"].update(
        {"fontSize": {"value": LABEL_SIZE}, "limit": {"value": 1400},
         "text": {"field": "display_label"}}
    )
    marks[9]["encode"]["enter"].update(
        {"size": {"value": 700}, "strokeWidth": {"value": 5}}
    )
    marks[10]["encode"]["enter"].update(
        {"fontSize": {"value": LABEL_SIZE}, "limit": {"value": 1400},
         "text": {"field": "display_label"}}
    )
    marks[11]["encode"]["enter"]["size"] = {"value": 26000}
    marks[12]["encode"]["enter"].update(
        {"dx": {"value": -135}, "dy": {"value": 115}, "fontSize": {"value": LABEL_SIZE}}
    )

    # Replace the line-swatch legend with node symbols.  Pulmonary symbols are
    # included only in the lung figure to keep the other legends compact.
    entries = [
        ("Arterial vessel", base.RED, "circle", False),
        ("Capillary", base.CAPILLARY_PURPLE, "circle", False),
        ("Sinusoid", base.BLUE, "circle", True),
        ("Venous vessel", base.BLUE, "circle", False),
        ("Portal system vessel", base.PORTAL_PURPLE, "circle", True),
    ]
    if model["organ_key"] == "lung":
        entries = [
            ("Systemic arterial vessel", base.RED, "circle", False),
            ("Systemic venous vessel", base.BLUE, "circle", False),
            ("Capillary", base.CAPILLARY_PURPLE, "circle", False),
            ("Pulmonary arterial vessel", base.BLUE, "square", False),
            ("Pulmonary venous vessel", base.RED, "square", False),
        ]
    legend = []
    for index, (label, color, shape, open_symbol) in enumerate(entries):
        legend.append(
            {
                "label": label,
                "color": color,
                "shape": shape,
                "fill": "#ffffff" if open_symbol else color,
                "stroke": color,
                "stroke_width": OPEN_NODE_STROKE if open_symbol else 2.0,
                "x": 2580,
                "y": 70 + index * 72,
            }
        )
    _data(spec, "legend")[:] = legend
    marks[13]["encode"]["enter"].update(
        {"x": {"value": 2510}, "y": {"value": 25}, "text": {"value": "Vessel class"},
         "fontSize": {"value": 40}}
    )
    marks[14] = {
        "type": "symbol",
        "from": {"data": "legend"},
        "encode": {"enter": {
            "x": {"field": "x"}, "y": {"field": "y"},
            "shape": {"field": "shape"}, "size": {"value": 700},
            "fill": {"field": "fill"}, "stroke": {"field": "stroke"},
            "strokeWidth": {"field": "stroke_width"},
        }},
    }
    marks[15]["encode"]["enter"].update(
        {"x": {"field": "x"}, "dx": {"value": 48}, "fontSize": {"value": 34}}
    )

    # The four extra legend marks were added by the previous reference spec.
    marks[16]["encode"]["enter"].update(
        {"x": {"value": 2560}, "x2": {"value": 2600},
         "y": {"value": 442}, "y2": {"value": 472}, "strokeWidth": {"value": 5}}
    )
    marks[17]["encode"]["enter"].update(
        {"x": {"value": 2628}, "y": {"value": 457}, "fontSize": {"value": 34}}
    )
    marks[18]["encode"]["enter"].update(
        {"x": {"value": 2580}, "y": {"value": 530}, "size": {"value": 700}}
    )
    marks[19]["encode"]["enter"].update(
        {"x": {"value": 2628}, "y": {"value": 530}, "fontSize": {"value": 34}}
    )
    if not SHOW_LEGEND:
        # Keep the complete legend specification and data in the JSON, but do
        # not show it on the per-organ figures.  Retaining the mark slots here
        # also preserves the Illustrator layer order expected by the exporter.
        # Set SHOW_LEGEND=True when the shared introductory page is prepared.
        for mark in marks[13:20]:
            mark["encode"]["enter"]["opacity"] = {"value": 0}
    return spec


def _physical_svg(svg: str) -> str:
    return re.sub(
        r'<svg\b([^>]*?)\bwidth="[^"]+"\s+height="[^"]+"',
        rf'<svg\1width="{CANVAS_WIDTH_IN}in" height="{CANVAS_HEIGHT_IN}in"',
        svg,
        count=1,
    )


def _strip_hidden_legend(svg: str) -> str:
    """Remove hidden legend artwork so it is not selectable in Illustrator."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    parents = {child: parent for parent in root.iter() for child in parent}
    for element in list(root.iter()):
        if element.get("opacity") == "0":
            parent = parents.get(element)
            if parent is not None:
                parent.remove(element)
    return ET.tostring(root, encoding="unicode")


def render_organ(csv_dir: Path, organ_key: str, json_dir: Path, svg_dir: Path, hra: dict):
    validated = base.build_organ_model(csv_dir, organ_key, hra)
    if not validated["validation"]["passed"]:
        raise RuntimeError(f"{organ_key}: {validated['validation']['errors']}")
    model = revision_layout(validated)
    spec = make_revision_spec(model)
    stem = f"{organ_key}_{OUTPUT_STEM}"
    json_path = json_dir / f"{stem}.json"
    svg_path = svg_dir / f"{stem}.svg"
    json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    svg = base.make_illustrator_friendly_svg(vlc.vega_to_svg(vg_spec=spec))
    svg = _strip_hidden_legend(svg)
    svg_path.write_text(_physical_svg(svg))
    return svg_path


def main():
    project = Path(
        "/Users/zeynepamac/Desktop/Butterfly_v2.5_Update/"
        "ftus_connected_by_vasculature_per_organ"
    )
    source = Path(__file__).resolve().parent
    hra = json.loads((source / "hra-asctb-all.v2.5.json").read_text())
    json_dir = project / "organs_reference_style_scientific_json_griffin_revision"
    svg_dir = project / "organs_reference_style_scientific_svg_griffin_revision"
    json_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    for organ_key in base.ORGAN_ORDER:
        path = render_organ(
            project / "organs_with_ftu_6_csv", organ_key, json_dir, svg_dir, hra
        )
        print(path)


if __name__ == "__main__":
    main()
