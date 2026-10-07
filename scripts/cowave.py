#!/usr/bin/env python3
"""Resolve named CoWave design points and render their catalog views.

The JSON catalog is the only source of design/configuration names.  ``point``
and ``build`` both use :func:`resolve_point`, so the machine-readable point and
the command that the build action executes cannot silently drift apart.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "designs/catalog.json"
DOCS = ROOT / "docs"
IMPLEMENTATIONS = DOCS / "implementations.md"
POINT_HTML = DOCS / "design-point-generator.html"
HTML_OUTPUT_DEFAULT = "/tmp/cowave-build"
PHASES = ("controller-xo", "compute-xo", "link", "host", "emconfig", "run", "all")
FIELDS = {
    "profile": "QUANTIZED_LAYER_PROFILE",
    "rms_lanes": "QUANTIZED_LAYER_RMS_LANES",
    "silu_lanes": "QUANTIZED_LAYER_SILU_LANES",
    "weight_read_outstanding": "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING",
    "attention_wave": "QUANTIZED_LAYER_ATTENTION_WAVE",
    "prefill_ffn_overlap": "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP",
}


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def catalog_identity(catalog):
    """Return the identity embedded in generated files and point JSON."""
    return hashlib.sha256(_canonical_json(catalog).encode("utf-8")).hexdigest()


def _repo_path(relative, label="path"):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Catalog {label} must be a non-empty repository-relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Catalog {label} must be repository-relative: {relative}")
    resolved = (ROOT / candidate).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"Catalog {label} escapes the repository: {relative}")
    return resolved


def _repo_relative(path):
    resolved = path.resolve()
    return "." if resolved == ROOT else resolved.relative_to(ROOT).as_posix()


def _href(relative):
    """Return a link from a generated page in ``docs/`` to a repo path."""
    target = _repo_path(relative)
    rel = os.path.relpath(target, DOCS).replace(os.sep, "/")
    if target.is_dir() and not rel.endswith("/"):
        rel += "/"
    return quote(rel, safe="/%:@-._~!$&'()*+,;=")


def _path_record(relative):
    target = _repo_path(relative)
    return {
        "path": relative,
        "href": _href(relative),
        "kind": "directory" if target.is_dir() else "file",
    }


def _workload_spec(design):
    spec = design.get("workload")
    if not isinstance(spec, dict):
        raise ValueError(f"Design {design['id']} is missing workload bounds")
    return spec


def _validate_workload_spec(design):
    spec = _workload_spec(design)
    required = ("prefill_tokens", "decode_tokens", "sequence_batch", "layers", "weights", "clock_mhz")
    for field in required:
        if field not in spec or not isinstance(spec[field], dict):
            raise ValueError(f"Design {design['id']} has incomplete workload bounds: {field}")
        value = spec[field]
        if field == "weights":
            choices = value.get("choices")
            if not isinstance(choices, list) or not choices or value.get("default") not in choices:
                raise ValueError(f"Design {design['id']} has invalid workload choices")
            continue
        if not isinstance(value.get("default"), int) or not isinstance(value.get("min"), int) or not isinstance(value.get("max"), int):
            raise ValueError(f"Design {design['id']} has non-integer workload bounds: {field}")
        if value["min"] > value["default"] or value["default"] > value["max"]:
            raise ValueError(f"Design {design['id']} has inconsistent workload bounds: {field}")


def _case_id(row):
    explicit = row.get("case_id")
    if explicit:
        return explicit
    precision = row.get("precision")
    if not precision:
        return None
    return f"{precision}-{'wave' if row.get('attention_wave') else 'ref'}"


def _summary_cases(summary_path):
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read evidence summary {summary_path}: {exc}") from exc
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Evidence summary has no cases array: {summary_path}")
    return summary, cases


def _validate_evidence(design, config):
    evidence = config.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Configuration {design['id']}/{config['name']} has no evidence paths")
    for index, relative in enumerate(evidence):
        path = _repo_path(relative, f"evidence[{index}]")
        if not path.exists():
            raise ValueError(f"Evidence path does not exist: {relative}")
        if path.is_dir() and not (path / "checksums.sha256").is_file():
            raise ValueError(f"Evidence package has no checksums.sha256: {relative}")

    summary_relative = config.get("evidence_summary")
    case_id = config.get("evidence_case")
    if bool(summary_relative) != bool(case_id):
        raise ValueError(f"Configuration {design['id']}/{config['name']} must pair evidence_summary and evidence_case")
    if not summary_relative:
        return
    summary_path = _repo_path(summary_relative, "evidence_summary")
    if not summary_path.is_file():
        raise ValueError(f"Evidence summary does not exist: {summary_relative}")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Evidence case must be a case ID: {design['id']}/{config['name']}")
    _, cases = _summary_cases(summary_path)
    matches = [row for row in cases if _case_id(row) == case_id]
    if len(matches) != 1:
        raise ValueError(f"Evidence case {case_id!r} is not a unique summary row in {summary_relative}")
    row = matches[0]
    if row.get("design") != design["id"] or row.get("config_name") != config["name"]:
        raise ValueError(
            f"Evidence case {case_id!r} does not select {design['id']}/{config['name']}"
        )
    expected = {
        "precision": design.get("precision_selector"),
        "profile": config.get("profile"),
        "rms_lanes": config.get("rms_lanes"),
        "silu_lanes": config.get("silu_lanes"),
        "outstanding": config.get("weight_read_outstanding"),
        "attention_wave": bool(config.get("attention_wave")),
        "prefill_ffn_overlap": bool(config.get("prefill_ffn_overlap")),
        "block_size": design.get("rows"),
        "peak_mac_per_cycle": design.get("peak_mac_per_cycle"),
    }
    for field, value in expected.items():
        if value is not None and row.get(field) != value:
            raise ValueError(
                f"Evidence case {case_id!r} disagrees with {design['id']}/{config['name']} field {field}"
            )


def validate_catalog(catalog):
    """Validate all links, names, geometry and evidence references."""
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("Unsupported or missing design catalog schema_version")
    designs = catalog.get("designs")
    if not isinstance(designs, list) or not designs:
        raise ValueError("Design catalog has no designs")
    seen_designs = set()
    for design in designs:
        if not isinstance(design, dict) or not isinstance(design.get("id"), str):
            raise ValueError("Every catalog design needs a string id")
        design_id = design["id"]
        if design_id in seen_designs:
            raise ValueError(f"Duplicate design id: {design_id}")
        seen_designs.add(design_id)
        for key in ("source", "guide", "recipe"):
            path = _repo_path(design.get(key), key)
            if not path.exists():
                raise ValueError(f"Catalog {key} does not exist for {design_id}: {design.get(key)}")
        if "source_manifest" in design:
            manifest = _repo_path(design["source_manifest"], "source_manifest")
            if not manifest.is_file():
                raise ValueError(f"Catalog source_manifest is not a file for {design_id}: {design['source_manifest']}")
        capabilities = design.get("capabilities")
        if not isinstance(capabilities, dict) or not isinstance(capabilities.get("point"), bool) or not isinstance(capabilities.get("build"), bool):
            raise ValueError(f"Design {design_id} needs point/build capabilities")
        _validate_workload_spec(design)
        if design.get("compute_cus") is not None:
            for key in ("compute_cus", "rows", "columns", "peak_mac_per_cycle"):
                if not isinstance(design.get(key), int) or design[key] <= 0:
                    raise ValueError(f"Design {design_id} has invalid geometry field {key}")
            if design["peak_mac_per_cycle"] != design["compute_cus"] * design["rows"] * design["columns"]:
                raise ValueError(f"Design {design_id} has inconsistent peak geometry")
        elif capabilities.get("geometry"):
            raise ValueError(f"Design {design_id} claims geometry capability without geometry fields")

        build = design.get("build", {})
        if capabilities["build"]:
            if not isinstance(build, dict) or not build.get("script"):
                raise ValueError(f"Build-capable design {design_id} has no build script")
            script = _repo_path(build["script"], "build script")
            if not script.is_file():
                raise ValueError(f"Build script does not exist for {design_id}: {build['script']}")
            if not isinstance(build.get("phases"), list) or not build["phases"]:
                raise ValueError(f"Build-capable design {design_id} has no phases")
            if not set(build["phases"]).issubset(PHASES):
                raise ValueError(f"Build phases are invalid for {design_id}")

        configs = design.get("configurations")
        if not isinstance(configs, list):
            raise ValueError(f"Design {design_id} configurations must be a list")
        config_names = set()
        for config in configs:
            if not isinstance(config, dict) or not isinstance(config.get("name"), str) or not config["name"]:
                raise ValueError(f"Design {design_id} has an invalid configuration")
            name = config["name"]
            if name in config_names:
                raise ValueError(f"Duplicate configuration {design_id}/{name}")
            config_names.add(name)
            if capabilities["build"]:
                if "precision_selector" not in design:
                    raise ValueError(f"Build-capable design {design_id} has no precision selector")
                missing = [field for field in FIELDS if field not in config]
                if missing:
                    raise ValueError(f"Configuration {design_id}/{name} is missing: {', '.join(missing)}")
            _validate_evidence(design, config)
    return catalog


def read_catalog(validate=True):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    return validate_catalog(catalog) if validate else catalog


def configuration_hash(design, config):
    """Identify selected architecture/knobs, not a source revision."""
    if config is None:
        return None
    data = {"design": design["id"], **{k: config[k] for k in FIELDS if k in config}}
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def source_identity(design):
    if "source_manifest" not in design:
        return None
    return hashlib.sha256(_repo_path(design["source_manifest"], "source_manifest").read_bytes()).hexdigest()


def select(catalog, design_id, config_name=None):
    designs = [d for d in catalog["designs"] if d["id"] == design_id]
    if not designs:
        raise ValueError("Unknown design; use 'list' for canonical names")
    design = designs[0]
    if config_name is None:
        return design, None
    configs = [c for c in design["configurations"] if c["name"] == config_name]
    if not configs:
        raise ValueError("Unknown configuration for " + design_id)
    return design, configs[0]


def _selected_workload(design, prefill=None, weights=None):
    spec = _workload_spec(design)
    prefill_spec = spec["prefill_tokens"]
    selected_prefill = prefill_spec["default"] if prefill is None else prefill
    if isinstance(selected_prefill, bool) or not isinstance(selected_prefill, int):
        raise ValueError("Prefill must be an integer")
    if not prefill_spec["min"] <= selected_prefill <= prefill_spec["max"]:
        raise ValueError(
            f"Prefill must be in {prefill_spec['min']}..{prefill_spec['max']}; leave room for one Decode token"
        )
    weight_spec = spec["weights"]
    selected_weights = weight_spec["default"] if weights is None else weights
    if selected_weights not in weight_spec["choices"]:
        raise ValueError("Weights must be one of: " + ", ".join(weight_spec["choices"]))
    return {
        "prefill_tokens": selected_prefill,
        "decode_tokens": spec["decode_tokens"]["default"],
        "sequence_batch": spec["sequence_batch"]["default"],
        "layers": spec["layers"]["default"],
        "weights": selected_weights,
        "clock_mhz": spec["clock_mhz"]["default"],
    }


def _build_script_command(design, phase):
    if phase not in PHASES:
        raise ValueError("Unknown build phase: " + phase)
    build = design.get("build", {})
    phases = build.get("phases", [])
    if phase not in phases:
        raise ValueError(f"Phase {phase} is not supported by {design['id']}")
    script = Path(build["script"])
    source = Path(design["source"])
    try:
        relative_script = script.relative_to(source)
    except ValueError as exc:
        raise ValueError(f"Build script for {design['id']} is outside its source root") from exc
    return ["bash", relative_script.as_posix(), phase]


def resolve(design, config, phase, output, prefill=None, weights=None):
    """Return the exact cwd, environment and phase command used by ``build``."""
    if not design.get("capabilities", {}).get("build") or "precision_selector" not in design:
        raise ValueError("Use the design's linked reproduction recipe for this implementation")
    if config is None:
        raise ValueError("A named configuration is required for a build-capable design")
    selected = _selected_workload(design, prefill, weights)
    precision = design["precision_selector"]
    env = {value: str(config[key]) for key, value in FIELDS.items()}
    artifact = Path(output).expanduser().resolve() / design["id"] / config["name"]
    prefix = "VITIS_QUANTIZED_" + precision.upper() + "_LAYER"
    env.update({prefix + "_ROOT": str(artifact), "TARGET": "hw_emu", "FREQUENCY": str(selected["clock_mhz"])})
    env["QUANTIZED_" + precision.upper() + "_LAYER_PREFILL"] = str(selected["prefill_tokens"])
    env["QUANTIZED_" + precision.upper() + "_LAYER_WEIGHTS"] = selected["weights"]
    env["QUANTIZED_" + precision.upper() + "_LAYER_BLOCK_SIZE"] = str(design["rows"])
    return ROOT / design["source"], env, _build_script_command(design, phase)


def _shell_command(env, command):
    return shlex.join(["env"] + [key + "=" + str(env[key]) for key in sorted(env)] + command)


def _conflicts(env):
    # Reject ambiguous inherited architecture settings, including internal
    # managed/default markers left by a previously sourced profile resolver.
    conflicts = [
        key for key, value in os.environ.items()
        if key.startswith(("QUANTIZED_", "VITIS_QUANTIZED_"))
        and (key not in env or value != env[key])
    ]
    conflicts += [key for key in ("TARGET", "FREQUENCY") if key in os.environ and os.environ[key] != env[key]]
    return sorted(conflicts)


def _evidence_point(design, config):
    packages = [_path_record(path) for path in config.get("evidence", [])] if config else []
    result = {"packages": packages, "case_id": None, "summary": None, "row": None}
    if not config or not config.get("evidence_summary"):
        return result
    summary_path = _repo_path(config["evidence_summary"], "evidence_summary")
    summary, cases = _summary_cases(summary_path)
    rows = [row for row in cases if _case_id(row) == config["evidence_case"]]
    if len(rows) != 1:
        raise ValueError(f"Evidence case {config['evidence_case']!r} is not a unique summary row")
    row = deepcopy(rows[0])
    result.update({
        "case_id": config["evidence_case"],
        "summary": _path_record(config["evidence_summary"]),
        "row": row,
        "summary_package": _path_record(config["evidence"][0]),
        "summary_status": summary.get("status"),
    })
    return result


def resolve_point(catalog, design_id, config_name=None, phase="host", output=None, prefill=None, weights=None):
    """Return a portable machine-readable design point."""
    validate_catalog(catalog)
    design, config = select(catalog, design_id, config_name)
    if not design.get("capabilities", {}).get("point", True):
        raise ValueError(f"Design does not support point generation: {design_id}")
    output = str(ROOT / "artifacts") if output is None else str(output)
    workload = _selected_workload(design, prefill, weights)
    geometry_keys = ("compute_cus", "rows", "columns", "peak_mac_per_cycle", "matrix_products_per_dsp")
    geometry = {key: design[key] for key in geometry_keys if key in design}
    paths = {
        "repo": _path_record("."),
        "source": _path_record(design["source"]),
        "guide": _path_record(design["guide"]),
        "recipe": _path_record(design["recipe"]),
    }
    if config:
        paths["evidence"] = [_path_record(path) for path in config.get("evidence", [])]
        if config.get("evidence_summary"):
            paths["evidence_summary"] = _path_record(config["evidence_summary"])
    if "source_manifest" in design:
        paths["source_manifest"] = _path_record(design["source_manifest"])
    identities = {
        "catalog_sha256": catalog_identity(catalog),
        "configuration_sha256": configuration_hash(design, config),
        "published_source_manifest_sha256": source_identity(design),
    }
    point_data = {
        "schema_version": 1,
        "design_id": design_id,
        "configuration_name": config["name"] if config else None,
        "design": deepcopy(design),
        "configuration": deepcopy(config),
        "capabilities": deepcopy(design.get("capabilities", {})),
        "geometry": geometry,
        "workload": {"bounds": deepcopy(_workload_spec(design)), "selected": workload},
        "paths": paths,
        "links": {
            "source": paths["source"]["href"],
            "guide": paths["guide"]["href"],
            "recipe": paths["recipe"]["href"],
        },
        "identities": identities,
        "configuration_sha256": identities["configuration_sha256"],
        "published_source_manifest_sha256": identities["published_source_manifest_sha256"],
        "evidence": _evidence_point(design, config),
    }
    point_data["links"]["repo"] = paths["repo"]["href"]
    if "source_manifest" in paths:
        point_data["links"]["source_manifest"] = paths["source_manifest"]["href"]
    if "evidence" in paths:
        point_data["links"]["evidence"] = [item["href"] for item in paths["evidence"]]
    if "evidence_summary" in paths:
        point_data["links"]["evidence_summary"] = paths["evidence_summary"]["href"]

    if design.get("capabilities", {}).get("build"):
        if config is None:
            raise ValueError("A named configuration is required for a build-capable design")
        cwd, env, command = resolve(design, config, phase, output, workload["prefill_tokens"], workload["weights"])
        phase_commands = {}
        for phase_name in design["build"]["phases"]:
            phase_argv = _build_script_command(design, phase_name)
            phase_commands[phase_name] = {
                "command": phase_argv,
                "shell_command": _shell_command(env, phase_argv),
            }
        point_data["build"] = {
            "supported": True,
            "phase": phase,
            "cwd": str(cwd),
            "cwd_relative": _repo_relative(cwd),
            "environment": env,
            "command": command,
            "phase_commands": phase_commands,
            "shell_command": _shell_command(env, command),
            "terminal_command": "cd " + shlex.quote(_repo_relative(cwd)) + " && " + _shell_command(env, command),
            "output_root": str(Path(output).expanduser().resolve()),
        }
        row = point_data["evidence"].get("row")
        if row:
            point_data["identities"]["evidence_source_manifest_sha256"] = row.get("source_manifest_sha256")
            point_data["identities"]["evidence_source_archive"] = row.get("source_archive")
    else:
        point_data["build"] = {
            "supported": False,
            "phase": None,
            "cwd": None,
            "cwd_relative": None,
            "environment": {},
            "command": None,
            "phase_commands": {},
            "shell_command": None,
            "terminal_command": None,
            "recipe_link": paths["recipe"]["href"],
            "reason": "This historical/component family has a linked recipe, not a unified catalog build resolver.",
        }
    return point_data


def _markdown_link(relative, label):
    return f"[{label}]({_href(relative)})"


def render(catalog):
    validate_catalog(catalog)
    identity = catalog_identity(catalog)
    source_identity_marker = " ".join(
        f"{design['id']}={source_identity(design) or 'none'}" for design in catalog["designs"]
    )
    lines = [
        "# CoWave design catalog", "",
        "<!-- Generated by scripts/cowave.py render; edit designs/catalog.json. -->",
        f"<!-- catalog_sha256={identity} -->", "",
        f"<!-- source_manifest_sha256: {source_identity_marker} -->", "",
        "[Documentation](README.md) | [Architecture](architecture.md) | [Evaluation](experiments.md) | [Setup](environment.md) | [Point generator](design-point-generator.html)", "",
        "## Read a design name", "",
        "`cowave-<arithmetic>-<compute CUs>-<logical rows>-<output columns>`", "",
        "For example, `cowave-int4-4-8-128` has four compute CUs, each with an 8×128 logical matrix array. Controller/status kernels are not counted. Rows are physical query capacity, not sequence batch or prompt length. Longer prompts are processed in blocks.", "",
        "`fix16` means signed fixed point, not IEEE FP16. `int4` and `int8` mean W4A4 and W8A8 matrix operands; vector, scaling and KV arithmetic are specified separately. DSP packing is already included in logical array capacity. It is not an extra throughput multiplier.", "",
        "## Designs and reproduction", "",
        "| Design | Logical MAC/cycle | Implementation scope | Capability | Read / reproduce |", "| --- | ---: | --- | --- | --- |",
    ]
    for design in catalog["designs"]:
        capability = "build + point" if design["capabilities"]["build"] else "point + recipe"
        lines.append(
            f"| `{design['id']}` | {design.get('peak_mac_per_cycle', 'Component-specific')} | {design['scope']} | {capability} | "
            f"[Design]({_href(design['guide'])}) · [Recipe]({_href(design['recipe'])}) · [Source]({_href(design['source'])}) |"
        )
    lines += [
        "", "## Named configurations", "",
        "A base name identifies geometry and arithmetic. A configuration adds mechanisms and service rates. Names are stable selections, not automatically the newest or fastest version. Exact source revisions and archived run identities remain separate.", "",
        "| Design | Configuration | RMS / SiLU lanes | Outstanding | Attention wave | Prefill FFN overlap | Evidence case | Evidence |", "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for design in catalog["designs"]:
        for config in design["configurations"]:
            evidence = " · ".join(_markdown_link(path, f"Package {index + 1}") for index, path in enumerate(config["evidence"]))
            case = config.get("evidence_case", "—")
            if config.get("evidence_summary"):
                case = f"`{case}` · {_markdown_link(config['evidence_summary'], 'summary')}"
            lines.append(
                f"| `{design['id']}` | `{config['name']}` | "
                f"{str(config.get('rms_lanes', '—')) + ' / ' + str(config.get('silu_lanes', '—'))} | "
                f"{config.get('weight_read_outstanding', '—')} | {config.get('attention_wave', '—')} | "
                f"{config.get('prefill_ffn_overlap', '—')} | {case} | {evidence} |"
            )
    lines += [
        "", "Quantized entries select the existing `integrated` profile plus the explicit overrides above. The frozen resolver, headers and per-case CFLAGS define the remaining parameters (FIFO depths, accumulators, packing, etc.); see the [source provenance and reproduction recipe](../cases/quantized-layer/README.md). The Fix16 entry groups resident-Q2.14 evidence with its individual historical source manifests; it does not imply that all runs used identical binaries.", "",
        "```bash", "python3 scripts/cowave.py list", "python3 scripts/cowave.py show cowave-int4-4-8-128 integrated-rms2-silu4-prefill-overlap", "python3 scripts/cowave.py point cowave-int4-4-8-128 integrated-rms2-silu4-prefill-overlap --prefill 66 --weights random --output /tmp/cowave-build", "python3 scripts/cowave.py build cowave-int4-4-8-128 integrated-rms2-silu4-prefill-overlap --phase host --dry-run", "```", "",
        "`point` emits validated links, geometry, workload bounds, identities and the exact quoted phase command. `build` consumes the same resolution. `build` uses HW Emu at a modeled 200 MHz, random weights, P66 and the design's physical block size. `--prefill`, `--weights` and `--output` change the workload/artifact destination, not the design name. Explicitly choose a configuration; no moving `latest` alias is used. See [usage](usage.md) for other execution paths.", "",
        "## Evidence and workload notation", "",
        "Every measurement must identify **design + configuration + source/binary identity + workload + timing scope**. The configuration SHA-256 shown by the tool hashes the architectural selection only; it is not a source hash or proof of validation.", "",
        "P66 is 66 prompt tokens, D1 is one real Decode query, B1 is one independent sequence, and L1 is one executed layer. P8/G2/L2 means an eight-token prompt and two sampled outputs across two layers: the prompt produces the first output, one D1 forward the second. Historical operator-sweep P1024 means only the last eight query rows against a long prefix, not a full 1024-token prefill. See [evaluation](experiments.md).", "",
        "CSim, finite-FIFO RTL CoSim, HW Emu and physical hardware are distinct evidence levels. Useful-MAC efficiency is not PE active time or FPGA resource utilization. A source release does not establish routed timing or checkpoint accuracy.", "",
        "## Legacy Label Migration", "", "Old names remain valid inside immutable results. New prose and figures use the names above; kernel symbols, operation IDs and historical manifests retain their original bytes.", "",
        "| Legacy alias | Interpretation |", "| --- | --- |",
    ]
    for design in catalog["designs"]:
        for alias in design["aliases"]:
            lines.append(f"| `{alias}` | `{design['id']}` |")
    for alias, meaning in catalog["legacy_context"]:
        lines.append(f"| `{alias}` | {meaning} |")
    lines += ["", "**Fix16 operator diagnostics** and **Small-shape protocol tests** describe evaluation scopes within a design, not extra accelerator generations. Software release numbers in CITATION.cff likewise do not identify hardware configurations.", ""]
    return "\n".join(lines)


def _html_json(value):
    return _canonical_json(value).replace("<", "\\u003c")


def _html_point_key(design_id, config_name):
    return design_id + "\n" + (config_name or "")


def _html_points(catalog):
    points = {}
    for design in catalog["designs"]:
        configs = design["configurations"] or [None]
        for config in configs:
            name = config["name"] if config else None
            point_data = resolve_point(
                catalog, design["id"], name, phase="host", output=HTML_OUTPUT_DEFAULT
            )
            # The standalone page must remain portable when served from a
            # different checkout.  The command and links are repo-relative;
            # the CLI JSON retains its exact absolute cwd.
            if point_data["build"]["supported"]:
                point_data["build"]["cwd"] = point_data["build"]["cwd_relative"]
            points[_html_point_key(design["id"], name)] = point_data
    return points


def render_html(catalog):
    validate_catalog(catalog)
    identity = catalog_identity(catalog)
    points = _html_points(catalog)
    catalog_json = _html_json(catalog)
    points_json = _html_json(points)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="catalog-sha256" content="{identity}">
<title>CoWave design point generator</title>
<style>
:root {{ color-scheme: light; font: 15px/1.45 system-ui, sans-serif; background: #f6f7f9; color: #18202a; }}
body {{ margin: 0; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 28px 20px 56px; }}
h1 {{ margin: 0 0 6px; font-size: 28px; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
.lede {{ margin: 0 0 24px; color: #4d5866; }}
.panel {{ background: #fff; border: 1px solid #d8dde5; border-radius: 8px; padding: 18px; margin: 14px 0; }}
.controls {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
label {{ display: grid; gap: 5px; color: #4d5866; font-size: 13px; }}
select, input {{ box-sizing: border-box; width: 100%; min-height: 36px; padding: 7px 9px; border: 1px solid #aeb8c5; border-radius: 4px; background: #fff; color: #18202a; font: inherit; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 5px 14px; margin: 0; }}
dt {{ color: #687483; }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
a {{ color: #075da8; }}
pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; background: #10151c; color: #e8edf4; border-radius: 5px; padding: 13px; }}
.muted {{ color: #687483; }}
.status {{ font-weight: 650; }}
@media (max-width: 760px) {{ .controls, .grid {{ grid-template-columns: 1fr; }} main {{ padding: 20px 14px 40px; }} }}
</style>
</head>
<body>
<main>
<h1>CoWave design point generator</h1>
<p class="lede">Select a catalog design and configuration to inspect its fixed geometry, workload bounds, provenance links, and reproducible phase command.</p>
<section class="panel controls" aria-label="Design point selection">
<label>Design<select id="design"></select></label>
<label>Configuration<select id="configuration"></select></label>
<label>Phase<select id="phase"></select></label>
<label>Prefill tokens<input id="prefill" type="number" min="1" step="1"></label>
<label>Weights<select id="weights"><option value="random">random</option><option value="zero">zero</option></select></label>
<label>Output root<input id="output" type="text" value="{HTML_OUTPUT_DEFAULT}"></label>
</section>
<section class="panel grid">
<div><h2>Selected point</h2><dl id="details"></dl></div>
<div><h2>Repository links</h2><dl id="links"></dl></div>
</section>
<section class="panel"><h2>Phase command</h2><p id="build-status" class="status"></p><pre id="command"></pre></section>
<section class="panel"><h2>Evidence</h2><dl id="evidence"></dl></section>
<p class="muted">Commands run from the repository root after environment setup. Archived measurements retain their original workload; changing a prompt length does not measure a new performance result.</p>
</main>
<script>
const CATALOG = {catalog_json};
const DEFAULT_POINTS = {points_json};
const DEFAULT_OUTPUT = {json.dumps(HTML_OUTPUT_DEFAULT)};
const PHASES = {json.dumps(list(PHASES))};
function clone(value) {{ return JSON.parse(JSON.stringify(value)); }}
function shellQuote(value) {{
  value = String(value);
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : "'" + value.replace(/'/g, "'\\\\''") + "'";
}}
function shellJoin(values) {{ return values.map(shellQuote).join(" "); }}
function pointKey(designId, configName) {{ return designId + "\\n" + (configName || ""); }}
function normalizedOutput(value) {{
  value = String(value || DEFAULT_OUTPUT).trim() || DEFAULT_OUTPUT;
  if (!value.startsWith("/")) throw new Error("Use an absolute output path in the browser selector");
  return value.replace(/\\/+$/, "") || "/";
}}
function pointFor(designId, configName, options) {{
  options = options || {{}};
  const base = DEFAULT_POINTS[pointKey(designId, configName || "")];
  if (!base) throw new Error("Unknown catalog design/configuration");
  const point = clone(base);
  const prefill = options.prefill_tokens ?? options.prefill ?? point.workload.selected.prefill_tokens;
  const weights = options.weights ?? point.workload.selected.weights;
  const bounds = point.workload.bounds.prefill_tokens;
  if (!Number.isInteger(prefill) || prefill < bounds.min || prefill > bounds.max) throw new Error("Prefill is outside catalog bounds");
  if (!point.workload.bounds.weights.choices.includes(weights)) throw new Error("Weights are outside catalog choices");
  point.workload.selected.prefill_tokens = prefill;
  point.workload.selected.weights = weights;
  if (!point.build.supported) return point;
  const output = normalizedOutput(options.output || point.build.output_root);
  const precision = point.design.precision_selector.toUpperCase();
  const phase = options.phase || point.build.phase;
  if (!point.build.phase_commands[phase]) throw new Error("Unknown build phase");
  point.build.phase = phase;
  point.build.command = point.build.phase_commands[phase].command;
  const artifact = (output === "/" ? "" : output) + "/" + point.design.id + "/" + point.configuration.name;
  point.build.output_root = output;
  point.build.environment["VITIS_QUANTIZED_" + precision + "_LAYER_ROOT"] = artifact;
  point.build.environment["QUANTIZED_" + precision + "_LAYER_PREFILL"] = String(prefill);
  point.build.environment["QUANTIZED_" + precision + "_LAYER_WEIGHTS"] = weights;
  const assignments = Object.keys(point.build.environment).sort().map(key => key + "=" + point.build.environment[key]);
  point.build.shell_command = shellJoin(["env"].concat(assignments, point.build.command));
  point.build.terminal_command = "cd " + shellQuote(point.build.cwd_relative) + " && " + point.build.shell_command;
  Object.keys(point.build.phase_commands).forEach(phase => {{
    const phasePoint = point.build.phase_commands[phase];
    phasePoint.shell_command = shellJoin(["env"].concat(assignments, phasePoint.command));
  }});
  return point;
}}
function link(label, href) {{ return href ? '<a href="' + href + '">' + label + '</a>' : '<span class="muted">—</span>'; }}
function setDetails(point) {{
  const geometry = Object.keys(point.geometry).map(key => key + "=" + point.geometry[key]).join(", ") || "component-level geometry";
  document.getElementById("details").innerHTML = [
    ["Design", point.design_id], ["Configuration", point.configuration_name || "recipe-only"],
    ["Geometry", geometry], ["Workload", point.build.supported ? "P" + point.workload.selected.prefill_tokens + " + D" + point.workload.selected.decode_tokens + ", B" + point.workload.selected.sequence_batch + ", L" + point.workload.selected.layers : "See the design-specific recipe"],
    ["Configuration SHA-256", point.configuration_sha256 || "—"], ["Catalog SHA-256", point.identities.catalog_sha256]
  ].map(row => '<dt>' + row[0] + '</dt><dd>' + row[1] + '</dd>').join("");
  document.getElementById("links").innerHTML = [
    ["Repository", point.links.repo], ["Source", point.links.source], ["Guide", point.links.guide], ["Recipe", point.links.recipe], ["Source manifest", point.links.source_manifest]
  ].map(row => '<dt>' + row[0] + '</dt><dd>' + link(row[0], row[1]) + '</dd>').join("");
  const evidence = point.evidence;
  document.getElementById("evidence").innerHTML = [
    ["Case", evidence.case_id || "—"], ["Summary", evidence.summary ? link("summary.json", evidence.summary.href) : "—"],
    ["Archived source", point.identities.evidence_source_manifest_sha256 || "—"],
    ["Packages", evidence.packages.map(item => link(item.path, item.href)).join(" · ") || "—"]
  ].map(row => '<dt>' + row[0] + '</dt><dd>' + row[1] + '</dd>').join("");
  const status = document.getElementById("build-status");
  status.textContent = point.build.supported ? "Build-supported phase: " + point.build.phase : "Recipe-only family; no unified catalog build command";
  document.getElementById("command").textContent = point.build.supported ? point.build.terminal_command : "Use the linked recipe: " + point.links.recipe;
}}
function populateConfigurations() {{
  const design = CATALOG.designs.find(item => item.id === document.getElementById("design").value);
  const select = document.getElementById("configuration");
  select.innerHTML = "";
  if (!design.configurations.length) select.add(new Option("recipe-only", ""));
  else design.configurations.forEach(item => select.add(new Option(item.name, item.name)));
  const base = DEFAULT_POINTS[pointKey(design.id, select.value)];
  const phase = document.getElementById("phase");
  phase.innerHTML = "";
  Object.keys(base.build.phase_commands).forEach(name => phase.add(new Option(name, name)));
  phase.value = "host";
  ["phase", "prefill", "weights", "output"].forEach(id => document.getElementById(id).disabled = !base.build.supported);
  const bounds = base.workload.bounds.prefill_tokens;
  const input = document.getElementById("prefill");
  input.min = bounds.min; input.max = bounds.max; input.value = bounds.default;
  document.getElementById("weights").value = base.workload.bounds.weights.default;
  update();
}}
function update() {{
  try {{
    const point = pointFor(document.getElementById("design").value, document.getElementById("configuration").value, {{
      phase: document.getElementById("phase").value, prefill: Number(document.getElementById("prefill").value), weights: document.getElementById("weights").value, output: document.getElementById("output").value
    }});
    setDetails(point);
  }} catch (error) {{ document.getElementById("command").textContent = error.message; }}
}}
if (typeof document !== "undefined") {{
  document.getElementById("design").innerHTML = CATALOG.designs.map(item => '<option value="' + item.id + '">' + item.id + '</option>').join("");
  document.getElementById("design").addEventListener("change", populateConfigurations);
  ["configuration", "phase", "prefill", "weights", "output"].forEach(id => document.getElementById(id).addEventListener("input", update));
  populateConfigurations();
}}
window.COWAVE = {{ catalog: CATALOG, points: DEFAULT_POINTS, pointFor, shellQuote, shellJoin, phases: PHASES }};
</script>
</body>
</html>
'''


def _render_point_format(point_data, output_format):
    if output_format == "json":
        return json.dumps(point_data, indent=2, sort_keys=True) + "\n"
    if output_format == "shell":
        if not point_data["build"]["supported"]:
            return "recipe=" + point_data["links"]["recipe"] + "\n"
        return point_data["build"]["terminal_command"] + "\n"
    lines = [
        f"# {point_data['design_id']}", "",
        f"Configuration: `{point_data['configuration_name'] or 'recipe-only'}`  ",
        f"Geometry: `{json.dumps(point_data['geometry'], sort_keys=True)}`  ",
        f"Workload: `{json.dumps(point_data['workload']['selected'], sort_keys=True)}`  ",
        f"Source: [{point_data['paths']['source']['path']}]({point_data['paths']['source']['href']})  ",
        f"Recipe: [{point_data['paths']['recipe']['path']}]({point_data['paths']['recipe']['href']})", "",
    ]
    if point_data["build"]["supported"]:
        lines += ["```bash", point_data["build"]["terminal_command"], "```", ""]
    else:
        lines += ["This family is recipe-only; use the linked reproduction recipe.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--check", action="store_true")
    point_parser = sub.add_parser("point")
    point_parser.add_argument("design")
    point_parser.add_argument("configuration", nargs="?")
    point_parser.add_argument("--phase", choices=PHASES, default="host")
    point_parser.add_argument("--output", default=str(ROOT / "artifacts"))
    point_parser.add_argument("--prefill", type=int, default=None)
    point_parser.add_argument("--weights", choices=["zero", "random"], default=None)
    point_parser.add_argument("--format", choices=["json", "shell", "markdown"], default="json")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("design")
    build_parser.add_argument("configuration")
    build_parser.add_argument("--phase", choices=PHASES, default="host")
    build_parser.add_argument("--output", default=str(ROOT / "artifacts"))
    build_parser.add_argument("--prefill", type=int, default=None)
    build_parser.add_argument("--weights", choices=["zero", "random"], default=None)
    build_parser.add_argument("--dry-run", action="store_true")
    show_parser = sub.add_parser("show")
    show_parser.add_argument("design")
    show_parser.add_argument("configuration", nargs="?")
    args = parser.parse_args()
    catalog = read_catalog()
    if args.action == "list":
        for design in catalog["designs"]:
            print(design["id"] + "\n  " + "\n  ".join(config["name"] for config in design["configurations"]))
        return
    if args.action == "render":
        markdown = render(catalog)
        page = render_html(catalog)
        if args.check:
            if not IMPLEMENTATIONS.is_file() or IMPLEMENTATIONS.read_text(encoding="utf-8") != markdown:
                raise ValueError("Catalog documentation is stale; run scripts/cowave.py render")
            if not POINT_HTML.is_file() or POINT_HTML.read_text(encoding="utf-8") != page:
                raise ValueError("Design point generator is stale; run scripts/cowave.py render")
            print("COWAVE CATALOG DOCUMENTATION PASS")
        else:
            IMPLEMENTATIONS.write_text(markdown, encoding="utf-8")
            POINT_HTML.write_text(page, encoding="utf-8")
        return
    if args.action == "show":
        design, config = select(catalog, args.design, args.configuration)
        result = {"design": design, "configuration": config}
        if config:
            result["configuration_sha256"] = configuration_hash(design, config)
        if "source_manifest" in design:
            result["published_source_manifest_sha256"] = source_identity(design)
        print(json.dumps(result, indent=2))
        return
    point_data = resolve_point(catalog, args.design, args.configuration, args.phase, args.output, args.prefill, args.weights)
    if args.action == "point":
        print(_render_point_format(point_data, args.format), end="")
        return
    if not point_data["build"]["supported"]:
        raise ValueError("Use the design's linked reproduction recipe for this implementation")
    conflicts = _conflicts(point_data["build"]["environment"])
    if conflicts:
        raise ValueError("Conflicting inherited settings; unset before selecting a named configuration: " + ", ".join(conflicts))
    print("configuration_sha256=" + str(point_data["configuration_sha256"]), flush=True)
    print("published_source_manifest_sha256=" + str(point_data["published_source_manifest_sha256"]), flush=True)
    print("cwd=" + point_data["build"]["cwd"], flush=True)
    print(point_data["build"]["shell_command"], flush=True)
    if not args.dry_run:
        cwd = Path(point_data["build"]["cwd"])
        subprocess.run(["bash", "scripts/verify_source.sh"], cwd=cwd, check=True)
        subprocess.run(point_data["build"]["command"], cwd=cwd, env={**os.environ, **point_data["build"]["environment"]}, check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
