"""Keep multi-agent poses separated so humanoids don't overlap in state or render.

Used by the symbolic coop world (walk_to / grab / carry_to) and the 3D renderer.
Agents sharing a point (e.g. co-carrying) are spread in a symmetric formation;
walk destinations that would land on another agent are nudged aside.
"""

from __future__ import annotations

import math

MIN_AGENT_SEP = 1.0  # m between agent base positions (~clear of G1 body width)


def formation_offsets(n: int, spacing: float = MIN_AGENT_SEP, axis: str = "x") -> list[tuple[float, float]]:
    """Symmetric offsets for n agents centered on (0, 0). axis='x' -> line along x."""
    if n <= 1:
        return [(0.0, 0.0)]
    span = (n - 1) * spacing
    start = -span / 2
    if axis == "x":
        return [(start + i * spacing, 0.0) for i in range(n)]
    return [(0.0, start + i * spacing) for i in range(n)]


def _rotate(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c - dy * s, dx * s + dy * c


def spread_agents(agents: dict[str, dict[str, float]], ids: list[str],
                  cx: float, cy: float, yaw: float | None = None,
                  spacing: float = MIN_AGENT_SEP) -> None:
    """Place agents in a line through (cx, cy), optionally aligned with yaw."""
    ids = sorted(ids)
    for aid, (dx, dy) in zip(ids, formation_offsets(len(ids), spacing)):
        if yaw is not None:
            dx, dy = _rotate(dx, dy, yaw)
        agents[aid]["x"] = cx + dx
        agents[aid]["y"] = cy + dy


def resolve_walk(agent_id: str, tx: float, ty: float,
                 agents: dict[str, dict[str, float]],
                 min_sep: float = MIN_AGENT_SEP) -> tuple[float, float]:
    """Nudge (tx, ty) so the agent does not end up on top of another agent."""
    tx, ty = float(tx), float(ty)
    for other_id, other in agents.items():
        if other_id == agent_id:
            continue
        dx, dy = tx - other["x"], ty - other["y"]
        d = math.hypot(dx, dy)
        if d >= min_sep:
            continue
        if d > 1e-6:
            s = min_sep / d
            tx, ty = other["x"] + dx * s, other["y"] + dy * s
        else:
            tx += min_sep
    return tx, ty


def spread_all(agents: dict[str, dict[str, float]], min_sep: float = MIN_AGENT_SEP) -> None:
    """Push apart any agents closer than min_sep (simple pairwise repulsion)."""
    ids = list(agents.keys())
    for _ in range(4):
        moved = False
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                dx = agents[b]["x"] - agents[a]["x"]
                dy = agents[b]["y"] - agents[a]["y"]
                d = math.hypot(dx, dy)
                if d >= min_sep or d < 1e-9:
                    continue
                push = (min_sep - d) / 2
                nx, ny = dx / d, dy / d
                agents[a]["x"] -= nx * push
                agents[a]["y"] -= ny * push
                agents[b]["x"] += nx * push
                agents[b]["y"] += ny * push
                moved = True
        if not moved:
            break
