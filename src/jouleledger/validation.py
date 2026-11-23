"""Integrity checks for results and for the code base."""
from __future__ import annotations

import ast
import gzip
import hashlib
from pathlib import Path
from typing import Any

import jsonschema

from .ledger import MODEL_TAG
from .telemetry import MEASURED_TAGS, NOT_MEASURED

FORBIDDEN_MODULES = frozenset({"pynvml", "cupy", "nvidia", "triton", "pycuda"})
VERDICTS = ["SUPPORTED", "REFUTED", "INCONCLUSIVE", "LIMITED_BY_INSTRUMENTATION", "NOT_EVALUATED"]
RUN_STATUS = ["NOT_RUN", "SMOKE_ONLY", "COMPLETE", "INVALID_SESSION"]
MEASURED_MODES = ("MEASURED_CPU_ENERGY", "MEASURED_GPU_ENERGY")
MODES = [None, *MEASURED_MODES, "CPU_RESOURCE_PROFILING", "MODEL_ESTIMATED_ENERGY", "CPU_SMOKE_TEST"]

METRICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["experiment_id", "experiment_name", "status", "execution_mode", "hardware", "energy_source", "measurement_boundary",
                 "dispersion_D", "D_ci", "ledger_coeffs", "ledger_holdout_mape", "ref_kernel_cv", "sampling_bias_table",
                 "hypothesis_verdict", "limitations"],
    "properties": {
        "experiment_id": {"const": "GAI-001"},
        "experiment_name": {"const": "JouleLedger"},
        "status": {"enum": RUN_STATUS},
        "execution_mode": {"enum": MODES},
        "hypothesis_verdict": {"enum": VERDICTS},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
}


class IntegrityError(ValueError):
    """Results break a research-integrity rule."""


def validate_metrics(metrics: dict[str, Any]) -> None:
    """Schema plus the rules a schema cannot express."""
    jsonschema.validate(metrics, METRICS_SCHEMA)
    source = metrics["energy_source"]
    measured = source in MEASURED_TAGS
    if metrics["execution_mode"] in MEASURED_MODES and not measured:
        raise IntegrityError("measured mode needs a physical source tag")
    if source == MODEL_TAG and metrics["execution_mode"] in MEASURED_MODES:
        raise IntegrityError("a modelled value is reported as measured")
    if measured and not metrics["measurement_boundary"]:
        raise IntegrityError("an energy value needs a measurement boundary")
    if not measured and metrics["dispersion_D"] is not None:
        raise IntegrityError("dispersion_D is an energy result and needs a physical source")
    if not measured and metrics["ledger_holdout_mape"] is not None:
        raise IntegrityError("ledger_holdout_mape is an energy result and needs a physical source")
    if metrics["hypothesis_verdict"] in ("SUPPORTED", "REFUTED"):
        if metrics["execution_mode"] not in MEASURED_MODES:
            raise IntegrityError("a hypothesis verdict needs physically measured energy in a full run")
        if not metrics.get("session_quality", {}).get("valid"):
            raise IntegrityError("a hypothesis verdict needs a valid session")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Every energy value in the raw data must carry its source, and no other row may have one."""
    for row in rows:
        has_energy = row.get("energy_total_j") not in (None, "")
        tag = row.get("source_tag")
        if has_energy and tag not in MEASURED_TAGS:
            raise IntegrityError(f"energy value tagged {tag}")
        if not has_energy and tag != NOT_MEASURED:
            raise IntegrityError(f"row tagged {tag} has no energy value")


def sha256_file(path: Path) -> str:
    """Hash decompressed content for gzip, because gzip headers vary between writes."""
    opener = gzip.open if path.suffix == ".gz" else open
    digest = hashlib.sha256()
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_gpu_usage(paths: list[Path]) -> list[str]:
    """Static check: no GPU module import and no `.cuda` attribute. gpu.py is the one permitted exception."""
    problems: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES):
                    problems.append(f"{path}:{node.lineno}: imports {name}")
            if isinstance(node, ast.Attribute) and node.attr == "cuda":
                problems.append(f"{path}:{node.lineno}: uses .cuda")
    return problems
