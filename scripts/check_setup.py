"""One-command setup check - run this FIRST on the eval machine.

Verifies the environment can run the LLM-task pipeline and says what's missing:

    python scripts/check_setup.py

Checks, in order:
  1. Python >= 3.12
  2. core imports: mujoco, numpy, Pillow, hud, fastmcp
  3. sim imports:  warp, newton, gizmo  (the physics engine)
  4. the tabletop-v1 scene exists
  5. one real rollout: the env serves the sim capability, the scripted agent
     pushes the mug, and the move-object task grades it (reward printed)

Exit code is 0 if every required check passes, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import warnings
from pathlib import Path

try:
    import authlib.deprecate  # noqa: F401
except ImportError:
    pass
warnings.filterwarnings("ignore", message=r"authlib\.jose module is deprecated")
try:
    import warp as _wp

    _wp.config.log_level = _wp.LOG_ERROR
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[bool, str, bool, str]] = []  # (ok, name, required, detail)


def record(ok: bool, name: str, detail: str = "", required: bool = True) -> bool:
    results.append((ok, name, required, detail))
    mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    tag = "" if required else " [optional]"
    print(f"  [{mark}] {name}{tag}" + (f" - {detail}" if detail else ""))
    return ok


def check_import(name: str, module: str, required: bool = True) -> bool:
    try:
        mod = __import__(module)
        return record(True, name, f"{module} {getattr(mod, '__version__', '')}".strip(), required)
    except Exception as exc:  # noqa: BLE001
        return record(False, name, f"`import {module}` failed: {type(exc).__name__}: {exc}", required)


async def rollout_check() -> bool:
    sys.path.insert(0, str(ROOT))
    from hud import LocalRuntime  # noqa: F401

    from examples.example_agent import ScriptedAgent
    from environment.env import move_object

    task = move_object(scene_id="tabletop-v1", target_object="mug",
                       goal_x=-0.2, goal_y=0.0, goal_z=0.75)
    job = await task.run(ScriptedAgent(), runtime=LocalRuntime(str(ROOT / "environment" / "env.py")))
    return record(job.reward > 0.5, "rollout: scripted agent scores move-object",
                  f"reward={job.reward:.4f}")


def main() -> int:
    print("=" * 70)
    print("Worldsim x HUD - setup check")
    print("=" * 70)

    print("\nPython")
    record(sys.version_info >= (3, 12), "Python >= 3.12", f"found {sys.version.split()[0]}")

    print("\nCore packages")
    core_ok = all([
        check_import("mujoco", "mujoco"),
        check_import("numpy", "numpy"),
        check_import("Pillow", "PIL"),
        check_import("hud", "hud"),
        check_import("fastmcp", "fastmcp"),
    ])

    print("\nPhysics engine (the local sim - bundled wheel)")
    sim_ok = all([
        check_import("warp", "warp"),
        check_import("newton", "newton"),
        check_import("gizmo", "gizmo"),
    ])

    print("\nDefault scene")
    scene_dir = ROOT / "scenes" / "tabletop-v1"
    scene_ok = record((scene_dir / "scene.xml").exists() and (scene_dir / "metadata.json").exists(),
                      "scene tabletop-v1 exists", str(scene_dir))

    print("\nRollout (serve sim + scripted agent + grade)")
    if core_ok and sim_ok and scene_ok:
        try:
            asyncio.run(rollout_check())
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(False, "rollout: scripted agent scores move-object", f"{type(exc).__name__}: {exc}")
    else:
        record(False, "rollout: scripted agent scores move-object", "skipped - a check above failed")

    failed = [name for ok, name, required, _ in results if required and not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"NOT READY - {len(failed)} required check(s) failed:")
        for name in failed:
            print(f"  - {name}")
        print("=" * 70)
        return 1
    print("READY. Next: hud eval environment/tasks.py claude --group 3")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
