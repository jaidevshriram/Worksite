"""Env-side `robot` bridge for the Franka/LIBERO VLA scene.

The *environment* serves observations over the `robot` (openpi/0) protocol and the
*agent* runs the policy and streams actions back. The framework owns the loop, the
wire codec, and telemetry; this bridge just owns the sim.

It drives the Newton sim (`sim/server.py` + `sim/control.py`): scene load, Franka
home pose, settle, and the OSC/IK control, exposed as `reset`/`step`/
`get_observation`. The agent gets raw `uint8` camera frames + the 8-dim state (the
openpi codec ships numpy directly - no base64).

Scoring is a graded signal - lift-progress shaping + binary success, not a plain
success/fail - exposed via `result()` and turned into subscores by the template
(see environment/vla_env.py).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# Software rendering by default so the sim renders on a CPU-only box (the cameras
# are rendered on the bridge's thread). Set before the first mujoco render context.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np

from hud.environment.robot import RobotBridge

# Importing the Newton sim runs the one-time Warp init and gives us the scene-load/
# step machinery + the Franka EE controller. We drive its module singleton (one scene
# per rollout process - the VLA path never shares it with the LLM `mcp` tool path,
# which lives in a different served env).
from sim import control as ctl
from sim import server as sim_server


class WorldsimFrankaBridge(RobotBridge):
    """Serve the franka-libero-v1 Newton scene over the `robot` protocol.

    One sim step per received action (OSC/IK + `decimation` Newton substeps).
    `result()` reports the graded lift score; the env template wraps it into subscores.
    """

    def __init__(
        self,
        *,
        decimation: int = 25,   # sim substeps per action: 25 = 20 Hz control at dt=0.002
        image_size: int = 256,  # LIBERO renders 256x256
        record_dir: str | None = None,  # set (or WORLDSIM_RECORD_DIR) to dump a dataset
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        super().__init__(host=host, port=port)
        self.decimation = decimation
        self.image_size = image_size
        self._record_dir = record_dir or os.environ.get("WORLDSIM_RECORD_DIR")
        # Episode/task state (set on reset).
        self._target_object = "block"
        self._instruction = ""
        self._lift_height = 0.55
        self._initial_z = 0.43
        self._final_z = 0.43
        self._max_steps = 200
        # Reused offscreen renderer (one per size; recreating per frame leaks GL).
        self._renderer: mujoco.Renderer | None = None
        # Recording (optional): the last obs sent, the current episode dir + step log.
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_state: list[float] = []
        self._ep_dir: Path | None = None
        self._ep_index = 0
        self._rec_t = 0
        self._step_log: Any = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def stop(self) -> None:
        await super().stop()
        if self._renderer is not None:  # close GL before interpreter teardown
            self._renderer.close()
            self._renderer = None

    async def reset(  # type: ignore[override]
        self,
        scene_id: str = "franka-libero-v1",
        target_object: str = "block",
        instruction: str = "pick up the red block",
        lift_height: float = 0.55,
        seed: int = 0,
        max_steps: int = 200,
    ) -> str:
        """Load the scene (home pose + settle), then return the instruction prompt.

        The base resets scoring + pushes the first frame around this - see RobotBridge.
        """
        result = sim_server.reset(
            scene_id=scene_id, seed=seed, max_episode_steps=max_steps
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"sim reset failed: {result['error']}")

        self._target_object = target_object
        self._instruction = instruction
        self._lift_height = lift_height
        self._max_steps = max_steps
        self._initial_z = self._object_z()
        self._final_z = self._initial_z
        # reset() builds a fresh MuJoCo model, so drop the renderer bound to the old one.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._start_recording_episode(seed)
        return instruction

    def step(self, action: np.ndarray) -> None:
        if self.terminated:
            return
        sim = sim_server._sim
        rmodel = sim.mj_model or sim.solver.mj_model
        rdata = sim.mj_data or sim.solver.mj_data
        with sim.lock:
            # OSC/IK once, held for `decimation` Newton substeps.
            ctrl = ctl.compute_ctrl(
                rmodel, rdata, np.asarray(action, dtype=float), sim.robot_idx, sim.ee_cfg
            ).tolist()
            for _ in range(self.decimation):
                sim_server._newton_step(ctrl)
            sim.step_count += 1
            sim._sync_render_data()

        self._record_step(action)
        self._final_z = self._object_z()
        # success = goal achieved (block lifted); a step-count timeout terminates the
        # episode but is NOT a success (spec: timeout != win).
        self.success = self._final_z >= self._lift_height
        self.total_reward = self._reward()
        self.terminated = self.success or sim.step_count >= self._max_steps

    def get_observation(self) -> tuple[dict[str, np.ndarray], bool] | None:
        sim = sim_server._sim
        if sim.newton_model is None:
            return None
        rmodel = sim.mj_model or sim.solver.mj_model
        rdata = sim.mj_data or sim.solver.mj_data
        with sim.lock:
            sim._sync_render_data()
            agentview = self._render("agentview", rmodel, rdata)
            wrist = self._render("wrist", rmodel, rdata)
            state = ctl.build_state(rmodel, rdata, sim.robot_idx)
        # Keys must equal the contract's observation feature leaf names.
        self._last_frames = {"observation/image": agentview, "observation/wrist_image": wrist}
        self._last_state = state.tolist()
        data = {**self._last_frames, "observation/state": state.astype(np.float32)}
        return data, self.terminated

    def result(self) -> dict[str, Any]:
        """Graded episode score (not binary).

        reward = 0.5*lift_progress + 0.5*success - self-consistent with the subscores
        the template emits (full lift scores 1.0; a partial lift gets partial credit).
        """
        progress = self._lift_progress()
        reward = self._reward()
        res = {
            "score": round(reward, 4),
            "success": bool(self.success),
            "total_reward": round(reward, 4),
            "lift_progress": round(progress, 4),
            "final_z": round(self._final_z, 4),
            "lift_height": self._lift_height,
        }
        self._finish_recording_episode(res)
        return res

    # ── grading helpers ──────────────────────────────────────────────────────
    def _lift_progress(self) -> float:
        span = self._lift_height - self._initial_z
        if span <= 1e-3:
            return 1.0
        return max(0.0, min(1.0, (self._final_z - self._initial_z) / span))

    def _reward(self) -> float:
        return 0.5 * self._lift_progress() + 0.5 * (1.0 if self.success else 0.0)

    def _object_z(self) -> float:
        obj = sim_server.get_object_state(object_name=self._target_object)
        if isinstance(obj, dict) and "error" not in obj:
            return float(obj["position"]["z"])
        return 0.0

    # ── rendering ────────────────────────────────────────────────────────────
    def _render(self, camera: str, rmodel: Any, rdata: Any) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(rmodel, height=self.image_size, width=self.image_size)
        cam = sim_server._resolve_camera(rmodel, camera)
        self._renderer.update_scene(rdata, cam)
        return np.ascontiguousarray(self._renderer.render(), dtype=np.uint8)  # HWC uint8

    # ── optional dataset recording (env-side, enable with WORLDSIM_RECORD_DIR) ──────
    def _start_recording_episode(self, seed: int) -> None:
        if not self._record_dir:
            return
        self._ep_dir = Path(self._record_dir) / f"episode_{self._ep_index:04d}"
        (self._ep_dir / "images").mkdir(parents=True, exist_ok=True)
        self._step_log = (self._ep_dir / "steps.jsonl").open("w")
        self._rec_t = 0

    def _record_step(self, action: np.ndarray) -> None:
        if not self._record_dir or self._ep_dir is None or not self._last_frames:
            return
        from PIL import Image

        t = self._rec_t
        Image.fromarray(self._last_frames["observation/image"]).save(self._ep_dir / "images" / f"agentview_{t:04d}.png")
        Image.fromarray(self._last_frames["observation/wrist_image"]).save(self._ep_dir / "images" / f"wrist_{t:04d}.png")
        self._step_log.write(json.dumps({"t": t, "state": self._last_state, "action": list(map(float, action))}) + "\n")
        self._rec_t = t + 1

    def _finish_recording_episode(self, res: dict[str, Any]) -> None:
        if not self._record_dir or self._ep_dir is None:
            return
        self._step_log.close()
        (self._ep_dir / "episode.json").write_text(json.dumps({
            "instruction": self._instruction, "target_object": self._target_object,
            "lift_height": self._lift_height, "steps": self._rec_t,
            "recorded_at": time.time(), **res,
        }, indent=2) + "\n")
        self._ep_index += 1
        self._ep_dir = None


__all__ = ["WorldsimFrankaBridge"]
