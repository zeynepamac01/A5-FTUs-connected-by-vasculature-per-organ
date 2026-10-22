"""Render the seven non-prostate Griffin revisions with clean cubic links.

The scientific model, layout coordinates, typography, node sizes, canvas, FTU
containers, and colors come directly from the approved Griffin/Katy revision.
Only level-changing vessel connectors use the smooth Bézier treatment approved
for the prostate figure.  The original revision outputs are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import organ_vasculature_biological_vega as base
import organ_vasculature_griffin_revision as revision


ORGANS = tuple(key for key in base.ORGAN_ORDER if key != "prostate")
CURRENT_ORGAN = ""


def _point_on_cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
    )


def _cubic_derivative(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        3 * u**2 * (p1[0] - p0[0])
        + 6 * u * t * (p2[0] - p1[0])
        + 3 * t**2 * (p3[0] - p2[0]),
        3 * u**2 * (p1[1] - p0[1])
        + 6 * u * t * (p2[1] - p1[1])
        + 3 * t**2 * (p3[1] - p2[1]),
    )


def _split_cubic(p0, p1, p2, p3):
    midpoint = lambda left, right: (
        (left[0] + right[0]) / 2,
        (left[1] + right[1]) / 2,
    )
    a, b, c = midpoint(p0, p1), midpoint(p1, p2), midpoint(p2, p3)
    d, e = midpoint(a, b), midpoint(b, c)
    f = midpoint(d, e)
    return (p0, a, d, f), (f, e, c, p3)


def _path(points):
    p0, p1, p2, p3 = points
    return (
        f"M{p0[0]:.2f},{p0[1]:.2f} "
        f"C{p1[0]:.2f},{p1[1]:.2f} "
        f"{p2[0]:.2f},{p2[1]:.2f} {p3[0]:.2f},{p3[1]:.2f}"
    )


def _cubic_geometry(p0, p1, p2, p3):
    first, second = _split_cubic(p0, p1, p2, p3)
    arrow = _point_on_cubic(p0, p1, p2, p3, 0.63)
    derivative = _cubic_derivative(p0, p1, p2, p3, 0.63)
    return {
        "path": _path((p0, p1, p2, p3)),
        "source_half_path": _path(first),
        "target_half_path": _path(second),
        "arrow_x": round(arrow[0], 2),
        "arrow_y": round(arrow[1], 2),
        "arrow_angle": round(math.degrees(math.atan2(derivative[1], derivative[0])) + 90, 2),
    }


def bezier_geometry(source: dict, target: dict, elbow_x=None, radius=36.0):
    """Return one restrained cubic with an exact two-color midpoint split."""
    del elbow_x, radius
    p0 = (float(source["x"]), float(source["y"]))
    p3 = (float(target["x"]), float(target["y"]))
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        path = f"M{p0[0]:.2f},{p0[1]:.2f}"
        return {
            "path": path,
            "source_half_path": path,
            "target_half_path": path,
            "arrow_x": round(p0[0], 2),
            "arrow_y": round(p0[1], 2),
            "arrow_angle": 0,
        }

    source_name = source.get("name", "").lower()
    target_name = target.get("name", "").lower()
    pair = (source_name, target_name)

    # Give the chamber-to-heart transition the same soft entrance used by the
    # approved prostate figure, even when the two nodes are nearly level.
    if source.get("type") == "heart" and abs(dy) < 10:
        p1 = (p0[0] + dx * 0.25, p0[1] - 78)
        p2 = (p3[0] - dx * 0.36, p3[1])
        return _cubic_geometry(p0, p1, p2, p3)

    # Small-intestine capillary loop: two distinct arcs, not two overlapping
    # straight lines.  The vertical supply and return also receive gentle bows.
    if CURRENT_ORGAN == "small_intestine":
        if pair == ("mucous plexus of small intestine", "branch of mucous plexus of small intestine"):
            return _cubic_geometry(p0, (p0[0] + dx * .32, p0[1] - 95),
                                   (p3[0] - dx * .32, p3[1] - 95), p3)
        if pair == ("branch of mucous plexus of small intestine", "mucous plexus of small intestine"):
            return _cubic_geometry(p0, (p0[0] + dx * .32, p0[1] + 95),
                                   (p3[0] - dx * .32, p3[1] + 95), p3)
        if pair == ("vasa recta of jejunum", "mucous plexus of small intestine"):
            return _cubic_geometry(p0, (p0[0] - 70, p0[1] + dy * .30),
                                   (p3[0] - 70, p3[1] - dy * .30), p3)
        if pair == ("mucous plexus of small intestine", "jejunal vein"):
            return _cubic_geometry(p0, (p0[0] + 70, p0[1] + dy * .30),
                                   (p3[0] + 70, p3[1] - dy * .30), p3)

    # Skin FTU loops use paired, separated arcs.  This removes the dark blob
    # created when outgoing and returning links occupied the same geometry.
    if CURRENT_ORGAN == "skin":
        loop_pairs = {
            ("deep cutaneous plexus", "branch of deep cutaneous plexus"): -55,
            ("branch of deep cutaneous plexus", "deep cutaneous plexus"): 55,
            ("subpapillary plexus", "branch of subpapillary plexus"): -55,
            ("branch of subpapillary plexus", "subpapillary plexus"): 55,
        }
        if pair in loop_pairs:
            bow = loop_pairs[pair]
            return _cubic_geometry(p0, (p0[0] + dx * .32, p0[1] + bow),
                                   (p3[0] - dx * .32, p3[1] + bow), p3)
        if pair in {
            ("cutaneous arteriole", "deep cutaneous plexus"),
            ("deep cutaneous plexus", "cutaneous venule"),
        }:
            side = -70
            return _cubic_geometry(p0, (p0[0] + side, p0[1] + dy * .30),
                                   (p3[0] + side, p3[1] - dy * .30), p3)

    # Mouth acini use a subtle two-arc lens around each capillary node, keeping
    # the three gland pathways readable and preventing the return paths from
    # visually merging with the arterial fan.
    if CURRENT_ORGAN == "mouth":
        if source_name.endswith("arteriole") and target_name.endswith("capillary"):
            return _cubic_geometry(p0, (p0[0] + dx * .35, p0[1] - 34),
                                   (p3[0] - dx * .35, p3[1] - 34), p3)
        if source_name.endswith("capillary") and target_name.endswith("venule"):
            return _cubic_geometry(p0, (p0[0] + dx * .35, p0[1] + 34),
                                   (p3[0] - dx * .35, p3[1] + 34), p3)

    if CURRENT_ORGAN == "kidney":
        if pair == ("renal efferent arteriole", "peritubular capillary"):
            return _cubic_geometry(
                p0,
                (p0[0] - 35, p0[1] + 330),
                (p3[0] + 315, p3[1]),
                p3,
            )
        if pair == ("peritubular capillary", "stellate venule of kidney"):
            return _cubic_geometry(
                p0,
                (p0[0] - 80, p0[1]),
                (p3[0] + 45, p3[1] + 80),
                p3,
            )

    if abs(dy) < 1e-6:
        p1 = (p0[0] + dx * 0.45, p0[1])
        p2 = (p3[0] - dx * 0.45, p3[1])
    elif abs(dx) < 1e-6:
        p1 = (p0[0], p0[1] + dy * 0.45)
        p2 = (p3[0], p3[1] - dy * 0.45)
    else:
        # Horizontal endpoint tangents create the broad, quiet arcs used in
        # the approved prostate reference without S-bends or extra elbows.
        p1 = (p0[0] + dx * 0.44, p0[1])
        p2 = (p3[0] - dx * 0.44, p3[1])

    return _cubic_geometry(p0, p1, p2, p3)


def _move_node(model, name, x=None, y=None):
    node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
    old_x, old_y = float(node["x"]), float(node["y"])
    new_x = old_x if x is None else float(x)
    new_y = old_y if y is None else float(y)
    node["x"], node["y"] = new_x, new_y
    node["label_x"] = float(node.get("label_x", old_x)) + new_x - old_x
    node["label_y"] = float(node.get("label_y", old_y)) + new_y - old_y


def _place_label(model, name, x, y, *, angle=-48, align="center", baseline="bottom"):
    node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
    node["label_x"] = float(x)
    node["label_y"] = float(y)
    node["label_angle"] = float(angle)
    node["label_align"] = align
    node["label_baseline"] = baseline


def _organ_specific_layout(model):
    organ = model["organ_key"]

    if organ == "large_intestine":
        _move_node(model, "branch of mucous plexus of colon", 2460, 700)

    elif organ == "skin":
        _move_node(model, "cutaneous arteriole", y=410)
        _move_node(model, "cutaneous venule", y=920)

    elif organ == "kidney":
        # Recreate the supplied nephron reference: the renal corpuscle sits at
        # upper right, the vasa-recta loop runs beneath it, and the
        # peritubular/stellate return forms a broad lower diagonal.
        arterial_names = [
            "left ventricle", "aorta", "ascending aorta", "aortic arch",
            "descending aorta", "descending thoracic aorta", "abdominal aorta",
            "renal artery", "segmental renal artery", "interlobar renal artery",
            "arcuate renal artery", "interlobular renal artery",
        ]
        arterial_x = [340, 440, 540, 640, 740, 840, 940, 1040, 1140, 1240, 1340, 1440]
        for name, x in zip(arterial_names, arterial_x):
            _move_node(model, name, x, 470)
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"], node["y"] - 20 - (arterial_names.index(name) % 2) * 34,
                         angle=-60, baseline="bottom")

        venous_positions = {
            "right atrium": (340, 760),
            "inferior vena cava": (520, 760),
            "renal vein": (700, 760),
            "segmental renal vein": (860, 760),
            "interlobar renal vein": (1020, 760),
            "arcuate renal vein": (1180, 760),
            "interlobular renal vein": (1340, 760),
            "straight venule of kidney": (1480, 600),
            "stellate venule of kidney": (1510, 870),
        }
        for name, (x, y) in venous_positions.items():
            _move_node(model, name, x, y)
        for index, name in enumerate([
            "right atrium", "inferior vena cava", "renal vein",
            "segmental renal vein", "interlobar renal vein",
            "arcuate renal vein", "interlobular renal vein",
        ]):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"], node["y"] + 42 + (index % 2) * 42,
                         angle=-52, baseline="top")

        kidney_nodes = {
            "renal afferent arteriole": (2470, 410),
            "glomerular capillary": (2800, 410),
            "renal efferent arteriole": (2700, 565),
            "descending vasa recta of kidney": (2660, 650),
            "vasa recta of kidney": (2220, 650),
            "ascending vasa recta of kidney": (1650, 650),
            "peritubular capillary": (1650, 1070),
        }
        for name, (x, y) in kidney_nodes.items():
            _move_node(model, name, x, y)

        kidney_labels = {
            "renal afferent arteriole": (2500, 365, 0, "start", "bottom"),
            "glomerular capillary": (2800, 365, 0, "center", "bottom"),
            "renal efferent arteriole": (2670, 550, 0, "end", "middle"),
            "descending vasa recta of kidney": (2620, 615, 0, "end", "bottom"),
            "vasa recta of kidney": (2220, 615, 0, "center", "bottom"),
            "ascending vasa recta of kidney": (1770, 615, 0, "start", "bottom"),
            "straight venule of kidney": (1450, 565, 0, "center", "bottom"),
            "peritubular capillary": (2200, 925, 0, "center", "middle"),
            "stellate venule of kidney": (1460, 930, 0, "center", "top"),
        }
        for name, (x, y, angle, align, baseline) in kidney_labels.items():
            _place_label(model, name, x, y, angle=angle, align=align,
                         baseline=baseline)

        overlays = {
            item["name"].replace("\n", " ").lower(): item
            for item in model["ftu_overlays"]
        }
        revision._set_box(overlays["nephron"], 1550, 2920, 170, 1160)
        revision._set_box(overlays["renal corpuscle"], 2410, 2890, 270, 490)

        # Place the nonvascular nephron structures directly on the pathways,
        # matching the supplied reference. Paired structures share a marker
        # and a complete single-line label.
        marker_specs = {
            "descending limb of loop of henle": (1650, 745, "thin descending limb"),
            "loop of henle ascending limb thin segment": (1950, 795, "thin ascending limb"),
            "outer medullary collecting duct": (
                2250, 845,
                "outer medullary collecting duct; inner medullary collecting duct",
            ),
            "inner medullary collecting duct": (2250, 845, ""),
            "thick ascending limb of loop of henle": (
                1650, 1070,
                "thick ascending limb; cortical collecting duct",
            ),
            "cortical collecting duct": (1650, 1070, ""),
        }
        marker_label_positions = {
            "descending limb of loop of henle": (1690, 745, "middle"),
            "loop of henle ascending limb thin segment": (1990, 795, "middle"),
            "outer medullary collecting duct": (1900, 865, "middle"),
            "inner medullary collecting duct": (1900, 865, "middle"),
            "thick ascending limb of loop of henle": (1700, 1120, "top"),
            "cortical collecting duct": (1700, 1120, "top"),
        }
        for name, (x, y, label) in marker_specs.items():
            overlay = overlays[name]
            revision._set_marker(overlay, x, y)
            overlay["display_label"] = label
            label_x, label_y, baseline = marker_label_positions[name]
            overlay["label_x"] = label_x
            overlay["label_y"] = label_y
            overlay["label_baseline"] = baseline

        for record in model.get("ftu_marker_links", []):
            record["path"] = ""
            record["stroke_opacity"] = 0.0

        # Keep the vessel names above the vasa-recta route and the FTU names
        # below it. One short perpendicular stem per FTU makes every
        # association explicit without turning the FTU squares into vessel
        # nodes or introducing crossing callout lines.
        upper_ftu_stems = {
            overlays["descending limb of loop of henle"]["id"]:
                "M1650.00,650.00 L1650.00,745.00",
            overlays["loop of henle ascending limb thin segment"]["id"]:
                "M1950.00,650.00 L1950.00,795.00",
            overlays["outer medullary collecting duct"]["id"]:
                "M2250.00,650.00 L2250.00,845.00",
        }
        connected_ftus = set()
        for record in model.get("ftu_marker_links", []):
            ftu_id = record.get("ftu_id")
            if ftu_id in upper_ftu_stems and ftu_id not in connected_ftus:
                record["path"] = upper_ftu_stems[ftu_id]
                record["stroke_opacity"] = 0.38
                connected_ftus.add(ftu_id)

        # The PDF draws one shared symbol for each paired label, so suppress
        # the duplicate symbol while retaining both underlying FTU records.
        shared_symbol_members = {
            overlays["inner medullary collecting duct"]["id"],
            overlays["cortical collecting duct"]["id"],
        }
        model["ftu_markers"] = [
            marker for marker in model.get("ftu_markers", [])
            if marker.get("id") not in shared_symbol_members
        ]

    elif organ == "lung":
        # Four compact, clearly separated lanes. Square nodes remain
        # intentional for the special pulmonary vessels; each arterial/venous
        # pair shares a tighter vertical interval.
        tracks = {
            "right ventricle": (340, 380),
            "pulmonary trunk": (520, 380),
            "left pulmonary artery": (700, 380),
            "segmental pulmonary artery": (880, 380),
            "subsegmental pulmonary artery": (1060, 380),
            "pulmonary arteriole": (1240, 380),
            "intra-acinar arteriole": (1450, 380),
            "pre-alveolar capillary": (1840, 380),
            "alveolar capillary": (2425, 460),
            "post-alveolar capillary": (1840, 540),
            "intra-acinar venule": (1450, 540),
            "pulmonary venule": (1240, 540),
            "subsegmental pulmonary vein": (1060, 540),
            "intersegmental pulmonary vein": (880, 540),
            "pulmonary vein": (700, 540),
            "left atrium": (520, 540),
            "left ventricle": (340, 810),
            "aorta": (500, 810),
            "ascending aorta": (660, 810),
            "aortic arch": (820, 810),
            "descending aorta": (980, 810),
            "descending thoracic aorta": (1140, 810),
            "bronchial artery": (1400, 810),
            "bronchial arteriole": (1840, 810),
            "bronchial capillary": (2425, 890),
            "bronchial venule": (1840, 970),
            "bronchial vein": (1400, 970),
            "accessory hemiazygos vein": (1120, 970),
            "azygos vein": (860, 970),
            "superior vena cava": (600, 970),
            "right atrium": (340, 970),
        }
        for name, (x, y) in tracks.items():
            _move_node(model, name, x, y)

        pulmonary_arterial_labels = [
            "right ventricle", "pulmonary trunk", "left pulmonary artery",
            "segmental pulmonary artery", "subsegmental pulmonary artery",
            "pulmonary arteriole",
        ]
        pulmonary_venous_labels = [
            "left atrium", "pulmonary vein", "intersegmental pulmonary vein",
            "subsegmental pulmonary vein", "pulmonary venule",
        ]
        systemic_arterial_labels = [
            "left ventricle", "aorta", "ascending aorta", "aortic arch",
            "descending aorta", "descending thoracic aorta", "bronchial artery",
        ]
        for index, name in enumerate(pulmonary_arterial_labels):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            lift = 26 + (index % 2) * 30
            _place_label(model, name, node["x"], node["y"] - lift,
                         angle=-52, baseline="bottom")

        for index, name in enumerate(pulmonary_venous_labels):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            drop = 38 + (index % 2) * 38
            _place_label(model, name, node["x"], node["y"] + drop,
                         angle=-52, baseline="top")

        for index, name in enumerate(systemic_arterial_labels):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            drop = 38 + (index % 2) * 38
            _place_label(model, name, node["x"], node["y"] + drop,
                         angle=-52, baseline="top")

        lower_track = [
            "right atrium", "superior vena cava", "azygos vein",
            "accessory hemiazygos vein", "bronchial vein",
        ]
        for index, name in enumerate(lower_track):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            drop = 48 + (index % 2) * 42
            _place_label(model, name, node["x"], node["y"] + drop,
                         angle=-52, baseline="top")

        # Kept here as documentation of the four lane families; all labels in
        # these lists are positioned above or below their own path only.
        _lane_labels = [
            *pulmonary_arterial_labels,
            *pulmonary_venous_labels,
            *systemic_arterial_labels,
            *lower_track,
        ]
        assert len(_lane_labels) == len(set(_lane_labels))

        horizontal_labels = {
            "intra-acinar arteriole": (1450, 422, "center"),
            "pre-alveolar capillary": (1770, 350, "end"),
            "post-alveolar capillary": (1770, 570, "end"),
            "intra-acinar venule": (1450, 580, "center"),
            "bronchial arteriole": (1770, 785, "end"),
            "bronchial venule": (1770, 1005, "end"),
            "alveolar capillary": (2425, 510, "center"),
            "bronchial capillary": (2425, 940, "center"),
        }
        for name, (x, y, align) in horizontal_labels.items():
            _place_label(model, name, x, y, angle=0, align=align,
                         baseline="middle")

    elif organ == "mouth":
        # Three quiet, parallel gland modules.  Their upstream branches fan
        # from the carotid trunk and their return branches converge toward the
        # jugular trunk in anatomical order, avoiding the tight central knot.
        mouth_positions = {
            "superficial temporal artery": (1250, 310),
            "superior parotid gland arteriole": (1806, 310),
            "superior parotid gland capillary": (2225, 355),
            "superior parotid gland venule": (1806, 400),
            "retromandibular vein": (1450, 400),
            "anterior branch of retromandibular vein": (1280, 465),
            "common facial vein": (1120, 530),
            "lingual artery": (1250, 610),
            "sublingual artery": (1500, 610),
            "sublingual gland arteriole": (1806, 610),
            "sublingual gland capillary": (2225, 655),
            "sublingual gland venule": (1806, 700),
            "sublingual vein": (1450, 700),
            "lingual vein": (1180, 760),
            "internal jugular vein": (900, 820),
            "facial artery": (1250, 820),
            "glandular branches of facial artery": (1500, 910),
            "submandibular gland arteriole": (1806, 910),
            "submandibular gland capillary": (2225, 955),
            "submandibular gland venule": (1806, 1000),
            "concomitant Wharton doct vein": (1500, 1000),
        }
        for name, (x, y) in mouth_positions.items():
            _move_node(model, name, x, y)

        upper_labels = [
            "superficial temporal artery", "lingual artery", "sublingual artery",
            "facial artery", "glandular branches of facial artery",
        ]
        for index, name in enumerate(upper_labels):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"], node["y"] - 38 - (index % 2) * 34,
                         angle=-55, baseline="bottom")

        return_labels = [
            "retromandibular vein", "anterior branch of retromandibular vein",
            "common facial vein", "sublingual vein", "lingual vein",
            "concomitant Wharton doct vein",
        ]
        for index, name in enumerate(return_labels):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"], node["y"] + 38 + (index % 2) * 38,
                         angle=-55, baseline="top")

        # Reserve a distinct pocket for every long label in the branch fan.
        # Horizontal labels are used where a complete inter-lane gap exists;
        # the remaining labels follow their own curve at a restrained angle.
        mouth_label_overrides = {
            "common carotid artery": (900, 520, -55, "center", "bottom"),
            "external carotid artery": (1040, 470, -55, "center", "bottom"),
            "retromandibular vein": (1510, 438, 0, "center", "top"),
            "anterior branch of retromandibular vein": (1325, 500, 0, "center", "top"),
            "common facial vein": (960, 700, 0, "center", "top"),
            "lingual artery": (1240, 575, 0, "center", "bottom"),
            "sublingual artery": (1510, 575, 0, "center", "bottom"),
            "sublingual vein": (1510, 742, 0, "center", "top"),
            "lingual vein": (1130, 755, 0, "center", "top"),
            "facial artery": (1260, 850, 0, "center", "top"),
            "glandular branches of facial artery": (1530, 868, -48, "center", "bottom"),
            "concomitant Wharton doct vein": (1500, 1045, 0, "center", "top"),
        }
        for name, (x, y, angle, align, baseline) in mouth_label_overrides.items():
            _place_label(model, name, x, y, angle=angle, align=align,
                         baseline=baseline)

    elif organ == "liver":
        # Keep the three liver flows visually distinct: systemic arterial
        # supply above, the pancreatic/portal chain through the middle, and
        # systemic venous return below.  The previous 47 px portal spacing
        # compressed fifteen named vessels into an unreadable knot.
        artery_names = [
            "left ventricle", "aorta", "ascending aorta", "aortic arch",
            "descending aorta", "descending thoracic aorta", "abdominal aorta",
            "celiac artery", "common hepatic artery", "proper hepatic artery",
            "left hepatic artery", "segmental hepatic artery",
        ]
        artery_x = [340, 450, 560, 670, 780, 890, 1000, 1110, 1250, 1385, 1520, 1655]
        for name, x in zip(artery_names, artery_x):
            _move_node(model, name, x, 430)

        for index, name in enumerate(artery_names):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            lift = 26 + (index % 2) * 38
            _place_label(model, name, node["x"], node["y"] - lift,
                         angle=-48, align="start", baseline="bottom")

        portal_names = [
            "splenic artery", "pancreatic artery", "interlobular artery of pancreas",
            "lobular artery of pancreas", "intralobular arteriole of pancreas",
            "lobular capillary of pancreas", "intralobular venule of pancreas",
            "lobular vein of pancreas", "interlobular vein of pancreas",
            "pancreatic vein", "splenic vein", "hepatic portal vein",
            "left hepatic portal vein", "left lateral hepatic portal vein",
            "lateral inferior segmental hepatic portal vein",
        ]
        portal_x = [
            1200, 1300, 1400, 1500, 1600, 1700,
            1780, 1860, 1940, 2020, 2100, 2180, 2260, 2340, 2420,
        ]
        for index, name in enumerate(portal_names):
            _move_node(model, name, portal_x[index], 730)
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"] - 100, 720,
                         angle=82, align="start", baseline="top")
        _place_label(
            model, "lateral inferior segmental hepatic portal vein",
            2100, 620, angle=0, align="center", baseline="middle",
        )

        for name, x in [
            ("segmental hepatic vein", 1655), ("left hepatic vein", 1120),
            ("inferior vena cava", 700), ("right atrium", 340),
        ]:
            _move_node(model, name, x, 1160)
        for index, name in enumerate([
            "right atrium", "inferior vena cava", "left hepatic vein",
            "segmental hepatic vein",
        ]):
            node = next(n for n in model["nodes"] if n.get("name", "").lower() == name.lower())
            _place_label(model, name, node["x"], 1192,
                         angle=0, align="center", baseline="top")

        # Open the lobule into its own right-hand region so its internal
        # convergence is legible and no longer competes with portal labels.
        overlays = {
            item["name"].replace("\n", " ").lower(): item
            for item in model["ftu_overlays"]
        }
        revision._set_box(overlays["liver lobule"], 2380, 2920, 250, 1200)
        _move_node(model, "hepatic arteriole", 2500, 475)
        _move_node(model, "hepatic portal venule", 2500, 725)
        _move_node(model, "hepatic sinusoid", 2780, 600)
        _move_node(model, "central vein of liver", 2780, 1040)
        _place_label(model, "hepatic arteriole", 2500, 430, angle=0,
                     align="center", baseline="bottom")
        _place_label(model, "hepatic portal venule", 2500, 775, angle=0,
                     align="center", baseline="top")
        _place_label(model, "hepatic sinusoid", 2780, 652, angle=0,
                     align="center", baseline="top")
        _place_label(model, "central vein of liver", 2780, 1090, angle=0,
                     align="center", baseline="top")

    nodes = {node["id"]: node for node in model["nodes"]}
    for link in model["links"]:
        link.update(bezier_geometry(nodes[link["source_id"]], nodes[link["target_id"]]))
    return model


def main():
    global CURRENT_ORGAN
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = Path(__file__).resolve().parent
    hra = json.loads((source / "hra-asctb-all.v2.5.json").read_text())
    json_dir = args.output / "json"
    svg_dir = args.output / "svg"
    json_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    original = revision._rounded_polyline_geometry
    original_layout = revision.revision_layout
    original_line_width = revision.LINE_WIDTH
    original_label_size = revision.LABEL_SIZE
    original_open_node_stroke = revision.OPEN_NODE_STROKE
    revision._rounded_polyline_geometry = bezier_geometry
    revision.revision_layout = lambda validated: _organ_specific_layout(original_layout(validated))
    try:
        for organ in ORGANS:
            CURRENT_ORGAN = organ
            if organ in {"lung", "kidney"}:
                revision.LINE_WIDTH = 1.5 * revision.PT_SCALE
                revision.LABEL_SIZE = 8 * revision.PT_SCALE
                revision.OPEN_NODE_STROKE = 1.5 * revision.PT_SCALE
            else:
                revision.LINE_WIDTH = original_line_width
                revision.LABEL_SIZE = original_label_size
                revision.OPEN_NODE_STROKE = original_open_node_stroke
            output = revision.render_organ(
                args.project / "organs_with_ftu_6_csv",
                organ,
                json_dir,
                svg_dir,
                hra,
            )
            if organ in {"lung", "kidney"}:
                svg = output.read_text()
                svg = svg.replace('font-family="Arial"', 'font-family="Benton Sans"')
                output.write_text(svg)
            print(output)
    finally:
        revision._rounded_polyline_geometry = original
        revision.revision_layout = original_layout
        revision.LINE_WIDTH = original_line_width
        revision.LABEL_SIZE = original_label_size
        revision.OPEN_NODE_STROKE = original_open_node_stroke


if __name__ == "__main__":
    main()
