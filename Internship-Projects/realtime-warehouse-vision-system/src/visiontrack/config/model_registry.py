from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visiontrack.config.settings import project_path


DEFAULT_CLASS_ALIASES = {
    "box": "box",
    "pallet": "pallet",
    "qr_code": "qr_code",
    "conveyor_belt": "conveyor_belt",
}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    path: Path
    conf: float = 0.20
    iou: float = 0.50
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    device: str | None = None
    description: str = ""
    class_aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_ALIASES))
    qr: dict[str, Any] = field(default_factory=dict)
    class_confidence: dict[str, float] = field(default_factory=dict)
    max_area_ratio: dict[str, float] = field(default_factory=dict)


def _profile_from_config(key: str, data: dict[str, Any], aliases: dict[str, str]) -> ModelProfile:
    return ModelProfile(
        key=key,
        path=project_path(data["path"]),
        conf=float(data.get("conf", 0.20)),
        iou=float(data.get("iou", 0.50)),
        imgsz=int(data.get("imgsz", 640)),
        tracker=str(data.get("tracker", "bytetrack.yaml")),
        device=data.get("device"),
        description=str(data.get("description", "")),
        class_aliases={**DEFAULT_CLASS_ALIASES, **aliases},
        qr=dict(data.get("qr", {})),
        class_confidence={
            str(name): float(value)
            for name, value in data.get("class_confidence", {}).items()
        },
        max_area_ratio={
            str(name): float(value)
            for name, value in data.get("max_area_ratio", {}).items()
        },
    )


def load_model_profiles(models_config: dict[str, Any]) -> dict[str, ModelProfile]:
    registry = models_config.get("model_registry", {})
    aliases = dict(registry.get("class_aliases", {}))
    profiles: dict[str, ModelProfile] = {}

    for key, data in registry.get("models", {}).items():
        if not data.get("enabled", True):
            continue
        profile = _profile_from_config(key, data, aliases)
        if profile.path.exists():
            profiles[key] = profile

    discovery_dir = project_path(registry.get("discovery_dir", "models"))
    if discovery_dir.exists():
        for model_path in sorted(discovery_dir.glob("*.pt")):
            key = model_path.stem
            if key not in profiles:
                profiles[key] = ModelProfile(
                    key=key,
                    path=model_path,
                    description="Auto-discovered trained model",
                    class_aliases={**DEFAULT_CLASS_ALIASES, **aliases},
                )

    return profiles


def choose_model_profile(
    models_config: dict[str, Any],
    requested_key: str | None = "auto",
) -> ModelProfile:
    profiles = load_model_profiles(models_config)
    if not profiles:
        raise FileNotFoundError("No trained model files were found under models/.")

    if requested_key and requested_key != "auto":
        normalized_key = _normalize_requested_model_key(requested_key)
        if normalized_key in profiles:
            return profiles[normalized_key]

        requested_path = project_path(requested_key)
        if requested_path.exists() and requested_path.suffix.lower() == ".pt":
            aliases = dict(models_config.get("model_registry", {}).get("class_aliases", {}))
            return ModelProfile(
                key=requested_path.stem,
                path=requested_path,
                description="Explicit trained model path",
                class_aliases={**DEFAULT_CLASS_ALIASES, **aliases},
            )

        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown model '{requested_key}'. Available: {available}")

    default_key = models_config.get("model_registry", {}).get("default_model_key", "auto")
    if default_key != "auto" and default_key in profiles:
        return profiles[default_key]

    return profiles[sorted(profiles)[0]]


def _normalize_requested_model_key(requested_key: str) -> str:
    path = Path(requested_key)
    if path.suffix.lower() == ".pt":
        return path.stem
    return requested_key
