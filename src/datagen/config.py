"""설정 파일 로딩. 여러 yaml을 순서대로 깊은 병합(뒤 파일이 우선)한다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(paths: list[Path]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_merge(cfg, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return cfg
