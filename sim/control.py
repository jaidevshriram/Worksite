"""LIBERO-style end-effector control + observation for a Franka Panda in MuJoCo.

This module is the bridge between a LIBERO-trained VLA (e.g.
``lerobot/pi05_libero_finetuned_v044``) and our MuJoCo/Newton sim. It is
deliberately **pure MuJoCo** (only ``numpy`` + ``mujoco``), so it runs without
Newton/Warp and wires into the Newton MCP server (``sim/server.py``) unchanged -
the server steps ``solver.mj_data``, which is a real ``mujoco.MjData``.

Contract reproduced (see scenes/franka-libero-v1/metadata.json):
  observation.state = [eef_pos(3), axisangle(eef_quat)(3), finger_qpos(2)]  -> 8
  action            = [d_eef_pos(3), d_eef_axisangle(3), gripper(1)]        -> 7
                      deltas are normalized OSC commands in ~[-1, 1]; gripper>0
                      closes (robosuite convention).

The action -> joint mapping is a resolved-rate (damped-least-squares IK)
approximation of robosuite's OSC_POSE controller. It is the main fidelity knob
for sim-to-sim transfer; the scale constants below mirror robosuite's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# ── LIBERO / robosuite OSC_POSE defaults ─────────────────────────────────────
# robosuite OSC_POSE maps a normalized action in [-1, 1] to a per-step delta via
# output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5] (m, m, m, rad, rad, rad).
POS_SCALE = 0.05
ROT_SCALE = 0.5

# World-frame translation mapping OUR sim frame (Franka base at the origin, table
# at +x) into LIBERO's frame, so observation.state lands in the distribution the
# checkpoint's MEAN_STD normalizer expects. Without it our eef x~0.55 normalizes to
# ~+5.7 std (LIBERO state mean x=-0.047, std=0.105 from HuggingFaceVLA/libero),
# corrupting the proprioceptive input. offset = libero_state_mean[:3] - home_eef:
#   home eef   = [ 0.555, 0.000, 0.514]
#   libero mean= [-0.047, 0.034, 0.765]
# Actions are deltas (translation-invariant) and the IK controller uses the raw sim
# eef pose, so ONLY the reported state shifts - arm motion is unaffected.
STATE_POS_OFFSET = np.array([-0.602, 0.034, 0.251], dtype=np.float32)

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
ARM_ACTUATORS = ("actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7")
GRIPPER_ACTUATOR = "actuator8"
EEF_SITE = "eef"

# Franka home configuration (Menagerie "home" keyframe); fingers open.
HOME_QPOS = {
    "joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": -1.57079,
    "joint5": 0.0, "joint6": 1.57079, "joint7": -0.7853,
    "finger_joint1": 0.04, "finger_joint2": 0.04,
}

# Gripper actuator ctrl: 0 = closed, 255 = open (0.04 m). See panda.xml actuator8.
GRIPPER_CTRL_CLOSED = 0.0
GRIPPER_CTRL_OPEN = 255.0


@dataclass
class EEControlConfig:
    """Tunable knobs for the OSC/IK controller (sim-to-sim transfer)."""

    pos_scale: float = POS_SCALE
    rot_scale: float = ROT_SCALE
    ik_damping: float = 0.05           # damped-least-squares lambda
    gripper_positive_closes: bool = True  # robosuite: action[6] > 0 -> close


@dataclass
class RobotIndex:
    """Cached MuJoCo ids/addresses for the Franka, resolved once per model."""

    eef_site: int
    arm_qadr: np.ndarray      # qpos addresses of the 7 arm joints
    arm_dofadr: np.ndarray    # dof (qvel) addresses of the 7 arm joints
    arm_ctrl: np.ndarray      # actuator ids for the 7 arm position servos
    arm_range: np.ndarray     # (7, 2) joint limits
    finger_qadr: np.ndarray   # qpos addresses of the 2 finger joints
    gripper_ctrl: int         # actuator id for the gripper

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "RobotIndex":
        def jid(name: str) -> int:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"joint '{name}' not found in model")
            return j

        def aid(name: str) -> int:
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                raise ValueError(f"actuator '{name}' not found in model")
            return a

        arm_j = [jid(n) for n in ARM_JOINTS]
        return cls(
            eef_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE),
            arm_qadr=np.array([model.jnt_qposadr[j] for j in arm_j]),
            arm_dofadr=np.array([model.jnt_dofadr[j] for j in arm_j]),
            arm_ctrl=np.array([aid(n) for n in ARM_ACTUATORS]),
            arm_range=np.array([model.jnt_range[j] for j in arm_j]),
            finger_qadr=np.array([model.jnt_qposadr[jid(n)] for n in FINGER_JOINTS]),
            gripper_ctrl=aid(GRIPPER_ACTUATOR),
        )


# ── quaternion / orientation helpers ─────────────────────────────────────────


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (position[3], quaternion[w,x,y,z]) of a site in the world frame."""
    pos = data.site_xpos[site_id].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return pos, quat


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> axis-angle 3-vector (axis * angle)."""
    res = np.zeros(3)
    # mju_quat2Vel with dt=1 yields the rotation vector (axis * angle).
    mujoco.mju_quat2Vel(res, np.asarray(quat, dtype=float), 1.0)
    return res


def axisangle_to_quat(vec: np.ndarray) -> np.ndarray:
    """Axis-angle 3-vector -> quaternion [w,x,y,z]."""
    vec = np.asarray(vec, dtype=float)
    angle = float(np.linalg.norm(vec))
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-9:
        axis = vec / angle
        mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """3D rotation taking current orientation to target (world frame)."""
    res = np.zeros(3)
    mujoco.mju_subQuat(res, np.asarray(target_quat, float), np.asarray(current_quat, float))
    return res


# ── home / observation / control ─────────────────────────────────────────────


def apply_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the arm + fingers to the Franka home pose and recompute kinematics."""
    for name, value in HOME_QPOS.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    # Initialize position-servo targets to the home pose so the arm holds still.
    idx = RobotIndex.from_model(model)
    home_arm = np.array([HOME_QPOS[n] for n in ARM_JOINTS])
    data.ctrl[idx.arm_ctrl] = home_arm
    data.ctrl[idx.gripper_ctrl] = GRIPPER_CTRL_OPEN
    mujoco.mj_forward(model, data)


def _compute_ori_offset_quat() -> np.ndarray:
    """Body-frame quaternion aligning our eef orientation convention to LIBERO's.

    Our Franka hand-mount + eef-site convention differs from robosuite's eef frame,
    so our gripper-down axis-angle (~[-2.215, -2.214, 0.016]) sits ~15 std from
    LIBERO's state orientation mean (~[2.972, -0.220, -0.126]). Map our home
    orientation onto LIBERO's with a fixed body-frame rotation
    R = q_home_ours^-1 ⊗ q_libero_mean, applied as reported = raw ⊗ R. Only the
    REPORTED state changes; the IK controller uses the raw site pose, so motion is
    unaffected (and pick orientations barely move anyway).
    """
    q_ours = axisangle_to_quat(np.array([-2.215, -2.214, 0.016]))
    q_lib = axisangle_to_quat(np.array([2.972, -0.220, -0.126]))
    q_ours_inv = np.zeros(4)
    mujoco.mju_negQuat(q_ours_inv, q_ours)
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, q_ours_inv, q_lib)
    return r


STATE_ORI_OFFSET_QUAT = _compute_ori_offset_quat()


def build_state(model: mujoco.MjModel, data: mujoco.MjData, idx: RobotIndex) -> np.ndarray:
    """LIBERO 8-dim proprioceptive state: [eef_pos(3), axisangle(3), fingers(2)].

    Position and orientation are mapped into LIBERO's frame (STATE_POS_OFFSET +
    STATE_ORI_OFFSET_QUAT) so the MEAN_STD-normalized state matches the checkpoint.
    """
    pos, quat = site_pose(model, data, idx.eef_site)
    rep_quat = np.zeros(4)
    mujoco.mju_mulQuat(rep_quat, quat, STATE_ORI_OFFSET_QUAT)
    axisangle = quat_to_axisangle(rep_quat)
    # robosuite/LIBERO reports the two finger qpos with OPPOSITE signs
    # (finger1 in [0,0.04], finger2 in [-0.04,0]); our MJCF has both positive.
    raw_fingers = data.qpos[idx.finger_qadr]
    fingers = np.array([raw_fingers[0], -raw_fingers[1]])
    return np.concatenate([pos + STATE_POS_OFFSET, axisangle, fingers]).astype(np.float32)


def gripper_ctrl_from_action(action_gripper: float, cfg: EEControlConfig) -> float:
    """Map LIBERO gripper action in [-1, 1] to actuator8 ctrl in [0, 255]."""
    g = float(np.clip(action_gripper, -1.0, 1.0))
    if not cfg.gripper_positive_closes:
        g = -g
    # g = +1 -> closed (0), g = -1 -> open (255).
    openness = (1.0 - g) / 2.0
    return GRIPPER_CTRL_OPEN * openness


def compute_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    action: np.ndarray,
    idx: RobotIndex,
    cfg: EEControlConfig,
) -> np.ndarray:
    """Resolved-rate IK: 7-dim delta-EE action -> full actuator ctrl vector.

    Returns a copy of ``data.ctrl`` with the 7 arm position targets and the
    gripper target overwritten. The caller writes this to ctrl and steps.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"expected 7-dim action, got {action.shape[0]}")

    cur_pos, cur_quat = site_pose(model, data, idx.eef_site)

    # Commanded world-frame deltas (OSC scaling).
    dpos = action[:3] * cfg.pos_scale
    drot = axisangle_to_quat(action[3:6] * cfg.rot_scale)
    target_quat = np.zeros(4)
    mujoco.mju_mulQuat(target_quat, drot, cur_quat)  # world-frame rotation

    pos_err = dpos
    ori_err = orientation_error(target_quat, cur_quat)
    err = np.concatenate([pos_err, ori_err])  # (6,)

    # Site Jacobian (6 x nv), restricted to the 7 arm dofs.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, idx.eef_site)
    J = np.vstack([jacp, jacr])[:, idx.arm_dofadr]  # (6, 7)

    lam2 = cfg.ik_damping ** 2
    dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), err)  # (7,)

    cur_arm = data.qpos[idx.arm_qadr]
    arm_target = np.clip(cur_arm + dq, idx.arm_range[:, 0], idx.arm_range[:, 1])

    ctrl = data.ctrl.copy()
    ctrl[idx.arm_ctrl] = arm_target
    ctrl[idx.gripper_ctrl] = gripper_ctrl_from_action(action[6], cfg)
    return ctrl
