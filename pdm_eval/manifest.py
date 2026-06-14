"""Manifest loading and cycle-artifact resolution for imported NiaNetVAE models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .detectors.imported_recurrent_autoencoder_detector import ARTIFACT_CONTRACT_VERSION


def _cycle_key(cycle_id: int) -> str:
    return f"{int(cycle_id):02d}"


def _load_cycle_manifest(path: str) -> tuple[dict, Path]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"PER_MAINT_MODEL_MANIFEST_PATH not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must use contract v2 "
            f"(schema_version={ARTIFACT_CONTRACT_VERSION!r}); "
            f"got schema_version={payload.get('schema_version')!r}."
        )
    if str(payload.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Imported cycle_manifest.json must declare "
            f"contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
            f"got contract_version={payload.get('contract_version')!r}."
        )
    if "cycles" not in payload or not isinstance(payload["cycles"], dict):
        raise ValueError(f"Invalid cycle manifest format at {p}: missing 'cycles' object.")
    for key, entry in payload["cycles"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cycle manifest format at {p}: cycle {key} entry is not an object.")
        status = str(entry.get("status", "")).strip().lower()
        if status == "trained":
            if str(entry.get("contract_version")) != ARTIFACT_CONTRACT_VERSION:
                raise ValueError(
                    f"Cycle {key} must use contract_version={ARTIFACT_CONTRACT_VERSION!r}; "
                    f"got {entry.get('contract_version')!r}."
                )
            missing = [field for field in ("model_path", "meta_path", "scaler_path") if not entry.get(field)]
            if missing:
                raise ValueError(
                    f"Cycle {key} status=trained is missing required v2 fields: {', '.join(missing)}."
                )
        elif status == "alias":
            if entry.get("alias_to") is None:
                raise ValueError(f"Cycle {key} status=alias is missing alias_to.")
        elif status != "missing":
            raise ValueError(f"Cycle {key} has unsupported status={status!r}.")
    return payload, p.parent


def _resolve_manifest_path(
    raw_path: Optional[str],
    manifest_dir: Path,
    cycle_id: Optional[int] = None,
) -> Optional[str]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return str((manifest_dir / candidate).resolve())

    if candidate.exists():
        return str(candidate.resolve())

    # Backward compatibility for old HPC-generated absolute Linux paths copied to another machine.
    fallbacks = []
    cycle_fragment = None
    for idx, part in enumerate(candidate.parts):
        if str(part).startswith("cycle_"):
            cycle_fragment = Path(*candidate.parts[idx:])
            break
    if cycle_fragment is not None:
        fallbacks.append((manifest_dir / cycle_fragment).resolve())

    if cycle_id is not None:
        fallbacks.append((manifest_dir / f"cycle_{_cycle_key(cycle_id)}" / candidate.name).resolve())

    fallbacks.append((manifest_dir / candidate.name).resolve())
    for fallback in fallbacks:
        if fallback.exists():
            return str(fallback)

    return str(candidate)


def _resolve_manifest_cycle(
    manifest: dict,
    manifest_dir: Path,
    cycle_id: int,
    strict: bool,
    visited: Optional[set] = None,
) -> Optional[dict]:
    visited = visited or set()
    key = _cycle_key(cycle_id)
    if key in visited:
        raise ValueError(f"Cycle alias loop detected while resolving cycle={key}.")
    visited.add(key)

    entry = manifest.get("cycles", {}).get(key)
    if entry is None:
        if strict:
            raise KeyError(f"Cycle {key} not present in manifest.")
        return None

    status = str(entry.get("status", "")).strip().lower()
    if status == "trained":
        model_path = _resolve_manifest_path(entry.get("model_path"), manifest_dir, cycle_id=int(cycle_id))
        meta_path = _resolve_manifest_path(entry.get("meta_path"), manifest_dir, cycle_id=int(cycle_id))
        scaler_path = _resolve_manifest_path(entry.get("scaler_path"), manifest_dir, cycle_id=int(cycle_id))
        if not model_path or not meta_path or not scaler_path:
            if strict:
                raise ValueError(f"Cycle {key} is trained but model_path/meta_path/scaler_path are missing.")
            return None
        model_exists = Path(model_path).exists()
        meta_exists = Path(meta_path).exists()
        scaler_exists = Path(scaler_path).exists()
        if strict and (not model_exists or not meta_exists or not scaler_exists):
            raise FileNotFoundError(
                f"Cycle {key} paths do not exist after resolution: "
                f"model_path={model_path} (exists={model_exists}), "
                f"meta_path={meta_path} (exists={meta_exists}), "
                f"scaler_path={scaler_path} (exists={scaler_exists})"
            )
        if not strict and (not model_exists or not meta_exists or not scaler_exists):
            return None
        resolved = dict(entry)
        resolved["resolved_cycle_id"] = int(entry.get("cycle_id", int(cycle_id)))
        resolved["model_path"] = model_path
        resolved["meta_path"] = meta_path
        resolved["scaler_path"] = scaler_path
        return resolved

    if status == "alias":
        alias_to = entry.get("alias_to")
        if alias_to is None:
            if strict:
                raise ValueError(f"Cycle {key} has status=alias but no alias_to.")
            return None
        return _resolve_manifest_cycle(
            manifest=manifest,
            manifest_dir=manifest_dir,
            cycle_id=int(alias_to),
            strict=strict,
            visited=visited,
        )

    if strict:
        raise ValueError(f"Cycle {key} unavailable in manifest (status={status!r}).")
    return None

