"""Compatibility shims for the bundled Newton wheel.

Imported for its side effects by `sim/server.py` before any solver is built.

----------------------------------------------------------------------------
color-31 collision-mask overflow
----------------------------------------------------------------------------
Newton's `SolverMuJoCo._convert_to_mjc` encodes per-shape collision filtering
into MuJoCo by graph-coloring every collidable shape and emitting
`contype = 1 << color`. It guards this with `if color < 32:` and otherwise
falls back to MuJoCo's default mask (collide-with-everything).

The off-by-one in that guard means a shape colored exactly **31** produces
`1 << 31 == 2147483648` (2^31), which is out of range for MuJoCo's int32
`MjsGeom.contype` and makes `MjsBody.add_geom` raise a TypeError. Scenes with
fewer than ~31 mutually-colliding shapes (e.g. tabletop-v1, franka-libero-v1)
never reach color 31, so this only bites dense scenes - notably Gizmo room
exports, which can have 150+ collidable meshes (>30 colors).

Fix: remap any color >= 31 to 32, dropping those shapes into the *exact same*
safe default-mask branch Newton already uses for colors 32+. Behaviorally this
is a no-op for colors 0..30 (filtering preserved); color-31 shapes simply
collide with everything (strictly more collision, never tunneling), identical
to how Newton already treats colors 32+.
"""

from __future__ import annotations

import numpy as np

_OVERFLOW_COLOR = 31  # 1 << 31 overflows int32 contype
_SAFE_DEFAULT_COLOR = 32  # >= 32 -> Newton uses MuJoCo's default collide-all mask


def _apply_color31_fix() -> None:
    try:
        from newton.solvers import SolverMuJoCo
    except Exception:
        return  # Newton/MuJoCo solver unavailable; nothing to patch.

    if getattr(SolverMuJoCo, "_worldsim_color31_patched", False):
        return

    # `_color_collision_shapes` is a @staticmethod(model, selected_shapes, ...);
    # accessing it on the class yields the plain underlying function.
    _orig = SolverMuJoCo._color_collision_shapes

    def _patched(*args, **kwargs):  # type: ignore[no-untyped-def]
        colors = _orig(*args, **kwargs)
        try:
            arr = np.asarray(colors)
            arr[arr >= _OVERFLOW_COLOR] = _SAFE_DEFAULT_COLOR
            return arr
        except Exception:
            return colors

    SolverMuJoCo._color_collision_shapes = staticmethod(_patched)
    SolverMuJoCo._worldsim_color31_patched = True


_apply_color31_fix()
