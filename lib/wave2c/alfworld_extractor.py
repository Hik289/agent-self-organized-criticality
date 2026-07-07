"""ALFWorld state extractor (§3.1) — 7-tuple z_{i,t} per step."""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict

LOOK_ACTIONS = frozenset({"look", "examine", "search"})
TAKE_ACTIONS = frozenset({"take", "pick"})
MUTATE_ACTIONS = frozenset({"put", "place", "heat", "cool", "clean", "slice", "toggle", "use", "open", "close"})

_RE_LOC = re.compile(r"\bin\s+the\s+(\w+)", re.IGNORECASE)
_RE_ON = re.compile(r"\bon\s+the\s+(\w+)", re.IGNORECASE)


@dataclass
class Z:
    q: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    c: dict = field(default_factory=dict)
    u: dict = field(default_factory=dict)
    r: dict = field(default_factory=dict)
    m: dict = field(default_factory=dict)
    p: str = ""

    def to_dict(self):
        return asdict(self)


def _extract_verb(action: str) -> str:
    tokens = (action or "").strip().split()
    return tokens[0].lower() if tokens else ""


def _tokens(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def extract_trajectory(traj: dict) -> list[dict]:
    step_events = traj["step_events"]
    receptacles_seen: set = set()
    objects_seen: set = set()
    rooms_visited: set = set()
    consecutive_look = 0
    examine_count = 0
    look_count = 0
    n_valid = 0
    n_fuzzy = 0
    unverified_mutations = 0
    mut_count = 0

    zs = []
    for step in step_events:
        step_i = step.get("step", 0)
        action = step.get("action_name", "") or ""
        env_reply = step.get("env_reply_head", "") or ""
        picked_type = step.get("picked_match_type", "")
        in_adm = bool(step.get("action_in_admissible", False))

        verb = _extract_verb(action)
        if picked_type == "fuzzy":
            n_fuzzy += 1
        if in_adm:
            n_valid += 1

        for loc in _RE_LOC.findall(env_reply):
            receptacles_seen.add(loc.lower())
        for loc in _RE_ON.findall(env_reply):
            receptacles_seen.add(loc.lower())
        for word in _tokens(env_reply):
            if len(word) >= 4 and word not in receptacles_seen:
                objects_seen.add(word)

        if "go" in verb or verb == "goto":
            rooms_visited.add(action.lower())

        if verb in LOOK_ACTIONS:
            consecutive_look += 1
            look_count += 1
            if "examine" in verb:
                examine_count += 1
        else:
            consecutive_look = 0

        if verb in MUTATE_ACTIONS:
            mut_count += 1
            recent_verbs = [_extract_verb(e.get("action_name", ""))
                            for e in step_events[max(0, step_i - 3):step_i]]
            if not any(v in LOOK_ACTIONS for v in recent_verbs):
                unverified_mutations += 1

        q = {"step_index": step_i, "actions_taken": step_i + 1, "in_admissible": in_adm}
        b = {"last_action": action, "last_env_reply_head": env_reply[:100],
             "n_admissible_shown": step.get("admissible_size_before", 0)}
        c = {"satisfied_count": n_valid, "violated_fuzzy_count": n_fuzzy,
             "unconfirmed": max(0, unverified_mutations)}
        u = {"consecutive_look": consecutive_look,
             "examine_ratio": examine_count / max(1, step_i + 1),
             "look_ratio": look_count / max(1, step_i + 1)}
        r = {"unverified_mutations": int(unverified_mutations),
             "n_mutating_actions_so_far": int(mut_count)}
        m = {"n_receptacles_seen": len(receptacles_seen),
             "n_objects_seen": len(objects_seen),
             "n_rooms_visited": len(rooms_visited)}
        zs.append(Z(q=q, b=b, c=c, u=u, r=r, m=m, p=verb).to_dict())
    return zs


def sigma_series_from_z(z_list: list[dict]) -> list[float]:
    out = []
    for z in z_list:
        U = float(z["u"].get("consecutive_look", 0))
        K = float(z["c"].get("violated_fuzzy_count", 0))
        R = 0.0
        V = float(z["r"].get("unverified_mutations", 0))
        B = float(max(0, z["b"].get("n_admissible_shown", 0) - 5))
        out.append(float(U + K + R + V + B))
    return out
