"""VLA agents for the `worldsim-vla` env's `robot` capability.

You implement the `Model` seam - `infer(batch) -> action chunk` - and the harness
(`RobotAgent`) owns connect/loop/chunking/telemetry. The env→policy wiring (which
camera is which, the state layout) comes from the env's contract at connect time, so
an agent carries no env-specific key names.

Four agents, in order of how much you write:

  * `PI05Agent`         - a stock LeRobot pi0.5 checkpoint, zero custom code (the baseline).
  * `CustomModel`/`CustomAgent` - bring your own policy: the SCAFFOLD below. Subclass
                          `Model`, implement `infer(batch) -> [N, T, A]` chunk.
  * `RemoteAgent`       - keep the weights on a remote GPU box (e.g. Modal) and run
                          the sim + loop on a CPU-only machine; only the stateless
                          observation→chunk forward crosses the network. Just
                          `RemoteModel` + `OpenPIAdapter`; serve the box from `serve/`
                          (drop your `infer` into `serve/policy_server.py` to serve it).
  * `NoopAgent`         - needs neither torch nor a GPU; verifies the wire end-to-end.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model

DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"


class _ReplanModel(Model):
    """LeRobot model that executes only the first `horizon` actions of each chunk.

    pi0.5 predicts a 50-action chunk; running it whole open-loop (~2.5 s blind) drifts
    in sim-to-sim transfer. The published LIBERO eval replans every ~10 - so we truncate
    each inferred chunk to `horizon`, and the harness re-infers once it is spent.
    """

    def __init__(self, policy: Any, preprocess: Any, postprocess: Any, *, horizon: int = 10) -> None:
        from hud.agents.robot.model import LeRobotModel

        self._inner = LeRobotModel(policy, preprocess, postprocess)
        self.horizon = horizon

    def reset(self) -> None:
        self._inner.reset()

    def infer(self, batch: Any) -> np.ndarray:
        # _inner.infer returns [N, T, A]; truncate the time axis T, keep the N dim
        # (ainfer indexes [0]; slicing the leading N axis would be a no-op for N=1).
        return self._inner.infer(batch)[:, : self.horizon]


class PI05Agent(RobotAgent):
    """Stock pi0.5 LIBERO checkpoint - the reference baseline (needs a GPU + lerobot)."""

    max_steps = 200

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        replan_horizon: int = 10,
    ) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        from hud.agents.robot.adapter import LeRobotAdapter

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[agent] loading policy: {checkpoint} (device={self.device})", flush=True)
        policy = PI05Policy.from_pretrained(checkpoint).to(self.device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.model = _ReplanModel(policy, preprocess, postprocess, horizon=replan_horizon)
        # Maps the env's two cameras onto the checkpoint's image slots in contract
        # order (the 3rd pi0.5 slot is auto zero-padded); state + prompt pass through.
        self.adapter = LeRobotAdapter(model_image_keys=list(policy.config.image_features))


class RemoteAgent(RobotAgent):
    """Run the policy on a remote GPU box; keep the sim + loop here (no local GPU).

    The GPU box serves the policy over the `openpi/0` websocket (see `serve/`), and
    only the stateless observation→chunk forward crosses the network. `RemoteModel`
    is the weightless client; `OpenPIAdapter` ships the env's `observation/*` frames
    as-is (the bridge already emits openpi wire keys), so there's nothing env-specific
    to configure here. The replan horizon lives server-side (the server truncates each
    chunk), so the harness re-infers exactly when the returned chunk is spent.
    """

    max_steps = 200

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        from hud.agents.robot.adapter import OpenPIAdapter
        from hud.agents.robot.model import RemoteModel

        self.model = RemoteModel(host, port)  # response_key="actions" (the serve/ default)
        self.adapter = OpenPIAdapter()


# ── SCAFFOLD: bring your own policy ───────────────────────────────────────────
# Copy these two classes, fill in the two TODOs, then run:
#   python run_vla.py --agent agents.vla_agent:CustomAgent --group 10
# To serve the SAME policy on a remote GPU box instead, mirror this `infer` in
# serve/policy_server.py (see build_pi05_infer) and use `run_vla.py --remote`.


class CustomModel(Model):
    """Your VLA policy. With the `OpenPIAdapter` (see `CustomAgent`) each inference
    receives the env's raw contract observation (no framework massaging):

        batch["observation/image"]        # HWC uint8 [256, 256, 3]  agentview camera
        batch["observation/wrist_image"]  # HWC uint8 [256, 256, 3]  wrist camera
        batch["observation/state"]        # float32  [8]   eef pose (pos+axis-angle) + gripper
        batch["prompt"]                   # str            the language instruction

    Return an action chunk shaped `[N, T, A]` - keep the leading `N=1` (the harness
    indexes `[0]`). `T` is your replan horizon (the loop re-infers once it's spent;
    ~10 is a good default for this scene). `A=7`: the ee-delta action
    `[dx, dy, dz, drx, dry, drz, gripper]`, gripper > 0 closes
    (units + ranges: contracts/franka_libero.json).
    """

    def __init__(self, horizon: int = 10) -> None:
        self.horizon = horizon
        # TODO: load your weights / processor once here (this runs at startup).
        # e.g. self.policy = MyPolicy.from_pretrained(...).to("cuda").eval()

    def infer(self, batch: Any) -> np.ndarray:
        # TODO: run your policy on `batch` -> `actions` of shape [T, 7] (T = self.horizon).
        # Placeholder (hold still, gripper open) so the scaffold runs before you wire a policy:
        actions = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (self.horizon, 1))
        return np.asarray(actions, dtype=np.float32)[None]  # [T, 7] -> [N=1, T, 7]


class CustomAgent(RobotAgent):
    """Your policy on the worldsim-vla env. `OpenPIAdapter` hands `infer` the raw contract
    observation above; for env-specific reshaping subclass `Adapter` instead."""

    max_steps = 200

    def __init__(self) -> None:
        from hud.agents.robot.adapter import OpenPIAdapter

        self.model = CustomModel()
        self.adapter = OpenPIAdapter()


# ──────────────────────────────────────────────────────────────────────────────


class NoopModel(Model):
    """Holds position, gripper open - proves the wire works with no GPU/model."""

    def infer(self, batch: Any) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)


class NoopAgent(RobotAgent):
    """Plumbing check: connect, observe, send a no-op every step. No torch required."""

    max_steps = 50
    adapter = None  # raw pass-through: the no-op ignores the observation

    def __init__(self) -> None:
        self.model = NoopModel()


__all__ = [
    "DEFAULT_CHECKPOINT", "CustomAgent", "CustomModel",
    "NoopAgent", "NoopModel", "PI05Agent", "RemoteAgent",
]
