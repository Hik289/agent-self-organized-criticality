from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number exceeds finite numeric range: {value}")
    return number


def read_input(path):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        value = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    value.append(json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float))
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    else:
        value = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    return value, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def records(value, key="rows"):
    result = value.get(key) if isinstance(value, dict) else value
    if not isinstance(result, list) or not result:
        raise ValueError(f"Input must contain a nonempty '{key}' array or a record array")
    if not all(isinstance(row, dict) for row in result):
        raise ValueError(f"Every '{key}' item must be an object")
    return result


def field(row, name):
    value = row
    for component in name.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"Missing required field '{name}'")
        value = value[component]
    return value


def write_output(path, result, sources, parameters):
    path = Path(path).expanduser().resolve()
    if any(path == Path(source["path"]) for source in sources):
        raise ValueError("Output must not overwrite an input file")
    payload = {
        "schema_version": 1,
        "sources": sources,
        "parameters": parameters,
        "result": result,
    }
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path
