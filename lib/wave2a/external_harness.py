from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import ModuleType


_LOAD_LOCK = threading.RLock()
_ENVIRONMENT_FILE = Path("anchor_setup/envs/statefulpuzzle_soc/env.py")
_JUDGE_FILES = (
    Path("anchor_3/state_extractor.py"),
    Path("anchor_3/local_judge.py"),
    Path("anchor_3/global_judge.py"),
)


class OriginalHarnessUnavailable(ImportError):
    pass


def _harness_root(required: tuple[Path, ...]) -> Path:
    configured = os.environ.get("AGENT_SOC_HARNESS_ROOT", "").strip()
    expected = ", ".join(str(path) for path in required)
    if not configured:
        raise OriginalHarnessUnavailable(
            "The original StatefulPuzzle/anchor_3 harness is not included in this release. "
            "Set AGENT_SOC_HARNESS_ROOT to its original experiments directory containing "
            f"{expected}. No replacement environment or scoring rules are substituted."
        )
    root = Path(configured).expanduser().resolve()
    missing = [str(root / path) for path in required if not (root / path).is_file()]
    if missing:
        raise OriginalHarnessUnavailable(
            "AGENT_SOC_HARNESS_ROOT is missing original harness files: " + ", ".join(missing)
        )
    return root


def _load_module(root: Path, relative: Path) -> ModuleType:
    source = root / relative
    digest = hashlib.sha256(str(source.parent).encode()).hexdigest()[:16]
    package_name = f"_agent_soc_original_harness_{digest}"
    module_name = f"{package_name}.{source.stem}"
    with _LOAD_LOCK:
        if module_name in sys.modules:
            return sys.modules[module_name]
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = [str(source.parent)]
            package.__package__ = package_name
            package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
            sys.modules[package_name] = package
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise OriginalHarnessUnavailable(
                f"Cannot import original harness module {source}: {exc}. "
                "Provide its original dependencies; sibling imports must be package-relative."
            ) from exc


def statefulpuzzle_types() -> tuple[type, type]:
    root = _harness_root((_ENVIRONMENT_FILE,))
    module = _load_module(root, _ENVIRONMENT_FILE)
    try:
        return module.StatefulPuzzleConfig, module.StatefulPuzzleSOC
    except AttributeError as exc:
        raise OriginalHarnessUnavailable(
            f"{root / _ENVIRONMENT_FILE} must define StatefulPuzzleConfig and StatefulPuzzleSOC."
        ) from exc


def anchor3_helpers():
    root = _harness_root(_JUDGE_FILES)
    modules = [_load_module(root, path) for path in _JUDGE_FILES]
    functions = []
    for module, path, name in zip(modules, _JUDGE_FILES, ("extract_trajectory", "judge_trajectory", "judge_trajectory")):
        function = getattr(module, name, None)
        if not callable(function):
            raise OriginalHarnessUnavailable(f"{root / path} must define callable {name}.")
        functions.append(function)
    return tuple(functions)


def require_statefulpuzzle_harness() -> None:
    _harness_root((_ENVIRONMENT_FILE, *_JUDGE_FILES))
    statefulpuzzle_types()
    anchor3_helpers()
