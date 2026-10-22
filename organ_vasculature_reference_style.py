from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import vl_convert as vlc

import organ_vasculature_biological_vega as base


OUTPUT_STEM = "vasculature_ftus_reference_style_scientific"


def _polyline_geometry(source: dict, target: dict, elbow_x: float | None = None) -> dict:
    """Build an orthogonal reference-style edge without changing endpoints."""
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

    lengths = [math.dist(left, right) for left, right in zip(points, points[1:])]
    total = sum(lengths)
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

    def path(values):
        return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in values)

    dx = arrow_right[0] - arrow_left[0]
    dy = arrow_right[1] - arrow_left[1]
    return {
        "path": path(points),
        "source_half_path": path(source_points),
        "target_half_path": path(target_points),
        "arrow_x": round(arrow[0], 2),
        "arrow_y": round(arrow[1], 2),
        "arrow_angle": round(math.degrees(math.atan2(dy, dx)) + 90, 2),
    }


def reference_layout(model: dict) -> dict:
    """Reposition a validated model in the supplied long-track visual system."""
    result = copy.deepcopy(model)
    old_height = float(model["height"])
    old_center = old_height / 2
    result["height"] = 900
    result["width"] = 3000
    # Retain enough of the authoritative route lanes to separate parallel
    # branches.  The three densest organs need extra separation so distinct
    # routes and their labels never collapse onto one another.  Kidney has
    # many more routes and its FTUs use a dedicated aligned column below.
    lane_scales = {"kidney": 0.48, "liver": 1.45, "mouth": 1.10, "lung": 1.10}
    lane_scale = lane_scales.get(result["organ_key"], 0.78)

    # Give the dense liver/mouth/lung tracks a little more of the fixed
    # 3000-pixel canvas.  Only geometry changes; node identity/order does not.
    dense_track = result["organ_key"] in {"liver", "mouth", "lung"}
    track_end = 2160 if dense_track else 2040
    target_end = 2600 if dense_track else 2540

    def map_x(value: float) -> float:
        value = float(value)
        if value <= 1410:
            return 240 + (value - 230) / 1180 * (track_end - 240)
        return track_end + (value - 1410) / 880 * (target_end - track_end)

    def map_target_y(value: float) -> float:
        return 475 + (float(value) - old_center) * 0.68

    for node in result["nodes"]:
        if node["type"] == "heart":
            node["x"], node["y"] = 120.0, 475.0
            continue
        old_y = float(node["y"])
        node["x"] = round(map_x(node["x"]), 2)
        if node.get("is_ftu_target"):
            node["y"] = round(map_target_y(old_y), 2)
            node["label_x"] = node["x"]
            node["label_y"] = node["y"] - 13
            node["label_baseline"] = "bottom"
            node["label_angle"] = 0
            node["label_align"] = "center"
        elif old_y <= old_center:
            node["y"] = round(300 + (old_y - (old_center - 210)) * lane_scale, 2)
        else:
            node["y"] = round(650 + (old_y - (old_center + 210)) * lane_scale, 2)
        node["shape"] = "circle"

    if result["organ_key"] == "liver":
        # The authoritative portal route branches from the celiac artery and
        # continues through the splenic/pancreatic circulation.  The generic
        # compression inherited the older drawing's x positions, which put
        # the splenic artery to the *left* of its celiac parent and made the
        # connecting elbow look detached.  Put this one directed chain in
        # source order; only x positions change.
        portal_route_order = [
            "splenic artery",
            "pancreatic artery",
            "interlobular artery of pancreas",
            "lobular artery of pancreas",
            "intralobular arteriole of pancreas",
            "lobular capillary of pancreas",
            "intralobular venule of pancreas",
            "lobular vein of pancreas",
            "interlobular vein of pancreas",
            "pancreatic vein",
            "splenic vein",
            "hepatic portal vein",
            "left hepatic portal vein",
            "left lateral hepatic portal vein",
            "lateral inferior segmental hepatic portal vein",
        ]
        portal_x0, portal_step = 1380.0, 70.0
        portal_x = {
            name: portal_x0 + index * portal_step
            for index, name in enumerate(portal_route_order)
        }
        for node in result["nodes"]:
            if node["type"] == "vessel" and node["name"].lower() in portal_x:
                node["x"] = portal_x[node["name"].lower()]

    if result["organ_key"] == "kidney":
        # Align each nephron-associated vessel with its FTU row.  This turns
        # the former diagonal association fan into short, readable horizontal
        # relationships while leaving the directed vascular graph unchanged.
        kidney_vessel_rows = {
            "renal afferent arteriole": 260.0,
            "glomerular capillary": 260.0,
            "renal efferent arteriole": 260.0,
            "ascending vasa recta of kidney": 330.0,
            "vasa recta of kidney": 400.0,
            "descending vasa recta of kidney": 435.0,
            "peritubular capillary": 470.0,
        }
        for node in result["nodes"]:
            row = kidney_vessel_rows.get(node["name"].lower())
            if row is not None:
                node["y"] = row

    # Keep target-vessel labels horizontal, but alternate levels whenever
    # several internal vessels share the same row (notably liver and kidney).
    target_rows = {}
    target_nodes = [
        node for node in result["nodes"]
        if node["type"] == "vessel" and node.get("is_ftu_target")
    ]
    for node in target_nodes:
        target_rows.setdefault(round(node["y"], 1), []).append(node)
    label_levels = [(-14, "bottom"), (15, "top"), (-32, "bottom"), (33, "top")]
    for row_nodes in target_rows.values():
        for index, node in enumerate(sorted(row_nodes, key=lambda item: item["x"])):
            offset, baseline = label_levels[index % len(label_levels)]
            node["label_x"] = node["x"]
            node["label_y"] = node["y"] + offset
            node["label_baseline"] = baseline
            node["label_angle"] = 0
            node["label_align"] = "center"

    # One shared baseline and angle along each long track. This mirrors the
    # reference figure and avoids the visual wobble of alternating tiers.
    vessels = [node for node in result["nodes"] if node["type"] == "vessel"]
    for upper in (True, False):
        track_nodes = sorted(
            (
                node for node in vessels
                if not node.get("is_ftu_target") and ((node["y"] < 475) == upper)
            ),
            key=lambda item: (item["x"], item["name"]),
        )
        for node in track_nodes:
            offset = 16
            node["label_x"] = node["x"]
            node["label_y"] = node["y"] - offset if upper else node["y"] + offset
            node["label_baseline"] = "bottom" if upper else "top"
            # A single, consistent angle per organ keeps labels aligned.
            # Dense tracks use a steeper angle so full authoritative names
            # remain readable without staggering them into arbitrary tiers.
            node["label_angle"] = -32 if dense_track else -15
            node["label_align"] = "center"

    for overlay in result["ftu_overlays"]:
        overlay["x"] = round(map_x(overlay["x"]), 2)
        overlay["x2"] = round(map_x(overlay["x2"]), 2)
        overlay["y"] = round(map_target_y(overlay["y"]), 2)
        overlay["y2"] = round(map_target_y(overlay["y2"]), 2)
        overlay["marker_x"] = round(map_x(overlay["marker_x"]), 2)
        overlay["marker_y"] = round(map_target_y(overlay["marker_y"]), 2)
        overlay["label_x"] = round(map_x(overlay["label_x"]), 2)
        overlay["label_y"] = round(map_target_y(overlay["label_y"]), 2)
        overlay["fill_opacity"] = 0.0
        overlay["stroke_opacity"] = min(float(overlay["stroke_opacity"]), 0.72)

    if result["organ_key"] == "kidney":
        # One clean right-side FTU column. Nephron children remain visibly
        # nested inside the nephron boundary; collecting ducts remain outside
        # that boundary, preserving the authoritative HRA hierarchy.
        kidney_rows = {
            "renal corpuscle": 260.0,
            "descending limb of loop of henle": 330.0,
            "loop of henle ascending limb thin segment": 400.0,
            "thick ascending limb of loop of henle": 470.0,
            "cortical collecting duct": 590.0,
            "inner medullary collecting duct": 650.0,
            "outer medullary collecting duct": 710.0,
        }
        for overlay in result["ftu_overlays"]:
            name = overlay["name"].lower()
            if name == "nephron":
                overlay.update(
                    {
                        "show_container": True,
                        "x": 2025.0,
                        "x2": 2940.0,
                        "y": 190.0,
                        "y2": 505.0,
                        "label_x": 2040.0,
                        "label_y": 215.0,
                        "label_baseline": "middle",
                        "stroke_opacity": 0.72,
                    }
                )
            elif name in kidney_rows:
                row_y = kidney_rows[name]
                marker_x = 2500.0 if row_y <= 505 else 2650.0
                overlay.update(
                    {
                        "show_container": False,
                        "marker_x": marker_x,
                        "marker_y": row_y,
                        "label_x": marker_x + 22.0,
                        "label_y": row_y,
                        "label_baseline": "middle",
                        "label_font_size": 14,
                    }
                )

    result["ftu_containers"] = [
        overlay for overlay in result["ftu_overlays"] if overlay["show_container"]
    ]
    result["ftu_markers"] = [
        overlay for overlay in result["ftu_overlays"] if not overlay["show_container"]
    ]
    node_lookup = {node["id"]: node for node in result["nodes"]}
    result["ftu_marker_links"] = []
    for overlay in result["ftu_markers"]:
        for vessel_id in overlay["member_vessel_ids"]:
            vessel = node_lookup[vessel_id]
            result["ftu_marker_links"].append(
                {
                    "vessel_id": vessel_id,
                    "ftu_id": overlay["id"],
                    "path": (
                        f"M{vessel['x']:.2f},{vessel['y']:.2f} "
                        f"L{overlay['marker_x']:.2f},{overlay['marker_y']:.2f}"
                    ),
                    "stroke_opacity": 0.30,
                }
            )

    if result["organ_key"] == "kidney":
        overlay_by_name = {item["name"].lower(): item for item in result["ftu_markers"]}
        vessel_by_name = {
            item["name"].lower(): item
            for item in result["nodes"]
            if item["type"] == "vessel"
        }
        pair_records = {
            (item["vessel_id"], item["ftu_id"]): item
            for item in result["ftu_marker_links"]
        }

        # Keep every validated vessel-to-FTU pair in Vega data, but draw each
        # multi-vessel association as one clean bracket/tree instead of
        # stacking duplicate paths on top of one another.
        for item in result["ftu_marker_links"]:
            item["path"] = ""
            item["stroke_opacity"] = 0.0

        def show(vessel_name: str, ftu_name: str, path: str):
            vessel = vessel_by_name[vessel_name]
            overlay = overlay_by_name[ftu_name]
            record = pair_records[(vessel["id"], overlay["id"])]
            record["path"] = path
            record["stroke_opacity"] = 0.30

        aff = vessel_by_name["renal afferent arteriole"]
        glo = vessel_by_name["glomerular capillary"]
        eff = vessel_by_name["renal efferent arteriole"]
        renal = overlay_by_name["renal corpuscle"]
        show(
            "renal afferent arteriole",
            "renal corpuscle",
            (
                f"M{aff['x']:.2f},244 L{eff['x']:.2f},244 "
                f"M{aff['x']:.2f},244 L{aff['x']:.2f},{aff['y']:.2f} "
                f"M{glo['x']:.2f},244 L{glo['x']:.2f},{glo['y']:.2f} "
                f"M{eff['x']:.2f},244 L{eff['x']:.2f},{eff['y']:.2f} "
                f"M{eff['x']:.2f},244 L2440,244 L2440,{renal['marker_y']:.2f} "
                f"L{renal['marker_x']:.2f},{renal['marker_y']:.2f}"
            ),
        )

        asc = vessel_by_name["ascending vasa recta of kidney"]
        desc = vessel_by_name["descending vasa recta of kidney"]
        desc_limb = overlay_by_name["descending limb of loop of henle"]
        inner = overlay_by_name["inner medullary collecting duct"]
        outer = overlay_by_name["outer medullary collecting duct"]
        show(
            "ascending vasa recta of kidney",
            "descending limb of loop of henle",
            (
                f"M{asc['x']:.2f},{asc['y']:.2f} L{desc_limb['marker_x']:.2f},{desc_limb['marker_y']:.2f} "
                f"M2530,{asc['y']:.2f} L2530,620 "
                f"M{desc['x']:.2f},{desc['y']:.2f} L2550,{desc['y']:.2f} L2550,620 "
                f"M2530,620 L2550,620 L2550,{inner['marker_y']:.2f} "
                f"L{inner['marker_x']:.2f},{inner['marker_y']:.2f} "
                f"M2550,{inner['marker_y']:.2f} L2550,{outer['marker_y']:.2f} "
                f"L{outer['marker_x']:.2f},{outer['marker_y']:.2f}"
            ),
        )

        vasa = vessel_by_name["vasa recta of kidney"]
        thin = overlay_by_name["loop of henle ascending limb thin segment"]
        show(
            "vasa recta of kidney",
            "loop of henle ascending limb thin segment",
            f"M{vasa['x']:.2f},{vasa['y']:.2f} L{thin['marker_x']:.2f},{thin['marker_y']:.2f}",
        )

        peri = vessel_by_name["peritubular capillary"]
        thick = overlay_by_name["thick ascending limb of loop of henle"]
        cortical = overlay_by_name["cortical collecting duct"]
        show(
            "peritubular capillary",
            "thick ascending limb of loop of henle",
            (
                f"M{peri['x']:.2f},{peri['y']:.2f} L{thick['marker_x']:.2f},{thick['marker_y']:.2f} "
                f"M2470,{peri['y']:.2f} L2470,{cortical['marker_y']:.2f} "
                f"L{cortical['marker_x']:.2f},{cortical['marker_y']:.2f}"
            ),
        )

        kidney_label_positions = {
            "renal afferent arteriole": (2091.14, 236.0, "bottom"),
            "glomerular capillary": (2176.36, 278.0, "top"),
            "renal efferent arteriole": (2261.59, 236.0, "bottom"),
            "ascending vasa recta of kidney": (2318.41, 316.0, "bottom"),
            "vasa recta of kidney": (2386.59, 386.0, "bottom"),
            "descending vasa recta of kidney": (2318.41, 449.0, "top"),
            "peritubular capillary": (2176.36, 484.0, "top"),
        }
        for node in result["nodes"]:
            values = kidney_label_positions.get(node["name"].lower())
            if values:
                node["label_x"], node["label_y"], node["label_baseline"] = values
                node["label_angle"] = 0
                node["label_align"] = "center"

    for link in result["links"]:
        source = node_lookup[link["source_id"]]
        target = node_lookup[link["target_id"]]
        # Make the liver portal branch visibly leave the celiac node itself.
        # This avoids a midpoint elbow that can look like a floating segment
        # when the figure is cropped or edited in Illustrator.
        elbow_x = None
        if (
            result["organ_key"] == "liver"
            and source["name"].lower() == "celiac artery"
            and target["name"].lower() == "splenic artery"
        ):
            elbow_x = source["x"]
        if (
            result["organ_key"] == "liver"
            and source["name"].lower() == "central vein of liver"
            and target["name"].lower() == "segmental hepatic vein"
        ):
            # Drain away from the central-vein node before turning toward the
            # heart.  The generic midpoint route ran backward through the
            # sinusoid and falsely suggested flow along the same line.
            elbow_x = source["x"]
        link.update(_polyline_geometry(source, target, elbow_x=elbow_x))

    # Layout must not alter the validated biological graph.
    assert {node["id"] for node in result["nodes"]} == {node["id"] for node in model["nodes"]}
    assert {
        (edge["source_id"], edge["target_id"]) for edge in result["links"]
    } == {
        (edge["source_id"], edge["target_id"]) for edge in model["links"]
    }
    return result


def make_reference_spec(model: dict) -> dict:
    spec = base.make_vega_spec(model)
    spec["description"] += "; reference-style scientific layout"
    spec["usermeta"]["layout_variant"] = "reference-style scientific; topology unchanged"

    # Compact reference-style key in the upper-left.
    legend_values = next(item["values"] for item in spec["data"] if item["name"] == "legend")
    for index, item in enumerate(legend_values):
        item["x"] = 18
        item["y"] = 25 + index * 23

    marks = spec["marks"]
    marks[0]["encode"]["enter"].update({"x": {"value": 260}, "y": {"value": 42}})
    marks[1]["encode"]["enter"].update({"x": {"value": 260}, "y": {"value": 67}})
    marks[3]["encode"]["enter"].update(
        {"strokeWidth": {"value": 0.9}, "strokeOpacity": {"field": "stroke_opacity"}}
    )
    marks[4]["encode"]["enter"]["strokeOpacity"] = {"field": "stroke_opacity"}
    marks[5]["encode"]["enter"].update(
        # Preserve directed edges in the validated Vega data, but do not draw
        # per-edge arrowheads: at publication scale they resemble extra nodes.
        {"size": {"value": 1}, "fillOpacity": {"value": 0}}
    )
    marks[6]["encode"]["enter"].update(
        {"shape": {"value": "circle"}, "size": {"value": 70}, "strokeWidth": {"value": 0.5}}
    )
    marks[7]["encode"]["enter"].update(
        {"fontSize": {"value": 12}, "fill": {"field": "color"}, "limit": {"value": 330}}
    )
    marks[8]["encode"]["enter"].update({"fontSize": {"field": "label_font_size"}})
    marks[11]["encode"]["enter"]["size"] = {"value": 7000}
    marks[12]["encode"]["enter"].update(
        {"dx": {"value": -82}, "dy": {"value": 4}, "fontSize": {"value": 18}}
    )
    marks[13]["encode"]["enter"].update(
        {"x": {"value": 18}, "y": {"value": 14}, "text": {"value": "Vessel class"}, "fontSize": {"value": 13}}
    )
    marks[14] = {
        "type": "rule",
        "from": {"data": "legend"},
        "encode": {"enter": {
            "x": {"field": "x"}, "x2": {"value": 68}, "y": {"field": "y"},
            "stroke": {"field": "color"}, "strokeWidth": {"value": 3},
        }},
    }
    marks[15]["encode"]["enter"].update(
        {"x": {"value": 78}, "dx": {"value": 0}, "fontSize": {"value": 12}}
    )
    marks.extend(
        [
            {
                "type": "rect",
                "encode": {"enter": {
                    "x": {"value": 28}, "x2": {"value": 58},
                    "y": {"value": 132}, "y2": {"value": 146},
                    "fill": {"value": "#ffffff"}, "stroke": {"value": base.INK},
                    "strokeWidth": {"value": 1.5},
                }},
            },
            {
                "type": "text",
                "encode": {"enter": {
                    "x": {"value": 78}, "y": {"value": 139},
                    "text": {"value": "FTU boundary (hollow)"},
                    "baseline": {"value": "middle"}, "font": {"value": "Arial"},
                    "fontSize": {"value": 12}, "fill": {"value": base.INK},
                }},
            },
            {
                "type": "symbol",
                "encode": {"enter": {
                    "x": {"value": 43}, "y": {"value": 163},
                    "shape": {"value": "triangle-up"}, "size": {"value": 150},
                    "fill": {"value": base.INK},
                }},
            },
            {
                "type": "text",
                "encode": {"enter": {
                    "x": {"value": 78}, "y": {"value": 163},
                    "text": {"value": "Heart"}, "baseline": {"value": "middle"},
                    "font": {"value": "Arial"}, "fontSize": {"value": 12},
                    "fill": {"value": base.INK},
                }},
            },
        ]
    )
    return spec


def render_organ(
    csv_dir: str | Path,
    organ_key: str,
    json_dir: str | Path,
    svg_dir: str | Path,
    hra_release: dict,
):
    json_dir, svg_dir = Path(json_dir), Path(svg_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    validated = base.build_organ_model(csv_dir, organ_key, hra_release)
    if not validated["validation"]["passed"]:
        raise RuntimeError(f"{organ_key}: validation failed: {validated['validation']['errors']}")
    model = reference_layout(validated)
    spec = make_reference_spec(model)
    stem = f"{organ_key}_{OUTPUT_STEM}"
    json_path = json_dir / f"{stem}.json"
    svg_path = svg_dir / f"{stem}.svg"
    json_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    svg = base.make_illustrator_friendly_svg(vlc.vega_to_svg(vg_spec=spec))
    svg_path.write_text(svg)
    return {"model": model, "spec": spec, "json": json_path, "svg": svg_path}


def render_all(csv_dir: str | Path, json_dir: str | Path, svg_dir: str | Path, hra_release: dict):
    results = {}
    for organ_key in base.ORGAN_ORDER:
        results[organ_key] = render_organ(csv_dir, organ_key, json_dir, svg_dir, hra_release)
    return results


if __name__ == "__main__":
    project = Path(
        "/Users/zeynepamac/Desktop/Butterfly_v2.5_Update/"
        "ftus_connected_by_vasculature_per_organ"
    )
    hra = json.loads((Path(__file__).resolve().parent / "hra-asctb-all.v2.5.json").read_text())
    rendered = render_all(
        project / "organs_with_ftu_6_csv",
        project / "organs_reference_style_scientific_json",
        project / "organs_reference_style_scientific_svg",
        hra,
    )
    for organ_key, paths in rendered.items():
        print(organ_key, paths["svg"])
