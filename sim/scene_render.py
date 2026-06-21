"""Offscreen 3D renderer for symbolic planning envs (no physics).

Supports multiple robot types (G1 humanoid, Spot quadruped) in the same scene.
Models are cached per (scene_id, agent_layout_key) so only the first call pays
the compile cost.

Public API
----------
get_renderer(scene_id, agent_types) -> SceneRenderer
    agent_types: list of robot type strings, e.g. ["g1","g1","spot","spot"]
    Omit agent_types to use the scene's default layout.

SceneRenderer.render(agents, objects, width, height) -> bytes
    agents: [(x, y, yaw), ...] one per agent in same order as agent_types
    objects: {body_name: (x, y, z, yaw)}  movable free bodies
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = Path(os.environ.get("SCENES_DIR", REPO_ROOT / "scenes"))

# ---------------------------------------------------------------------------
# Robot type registry
# ---------------------------------------------------------------------------
# G1 standing keyframe joints (indices 7.. of key_qpos[0]):
# elbow/shoulder angles that give a natural standing pose.
_G1_STAND = [
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0.2, 0.2, 0., 1.28, 0., 0., 0., 0.2, -0.2, 0., 1.28, 0., 0., 0.
]
_SPOT_STAND = [0., 1.04, -1.8, 0., 1.04, -1.8, 0., 1.04, -1.8, 0., 1.04, -1.8]

# Walk-cycle angular velocity: radians per frame at 30 fps ≈ 1.3 cycles/sec
# (one full step cycle every ~23 frames — comfortable walking cadence).
WALK_ANGULAR_VEL: float = 2.0 * math.pi / 23


def _g1_walk_joints(phase: float) -> list[float]:
    """G1 bipedal walking joint angles at the given gait phase (0 → 2π = one cycle).

    Joint order (29 values after the 7-DOF floating base):
      0  left_hip_pitch      6  right_hip_pitch     12  waist_yaw
      1  left_hip_roll       7  right_hip_roll      13  waist_roll
      2  left_hip_yaw        8  right_hip_yaw       14  waist_pitch
      3  left_knee           9  right_knee          15-21 left arm
      4  left_ankle_pitch   10  right_ankle_pitch   22-28 right arm
      5  left_ankle_roll    11  right_ankle_roll
    """
    φ = phase
    π = math.pi

    # ── leg parameters ──────────────────────────────────────────────
    HIP_AMP   = 0.38   # hip pitch swing ±rad
    KNEE_BASE = 0.08   # slight resting bend
    KNEE_AMP  = 0.55   # extra bend at peak swing
    ANKLE_AMP = 0.22   # ankle plantarflexion / dorsiflexion
    HIP_ROLL  = 0.06   # small lateral sway

    # Left leg leads by 0; right leg is π out of phase.
    # Hip pitch: negative = leg sweeps forward.
    l_hip_pitch  = -HIP_AMP * math.sin(φ)
    r_hip_pitch  = -HIP_AMP * math.sin(φ + π)

    # Knee bends during the swing phase (only on the upswing half of the cycle).
    l_knee = KNEE_BASE + KNEE_AMP * max(0.0, math.sin(φ + π * 0.25))
    r_knee = KNEE_BASE + KNEE_AMP * max(0.0, math.sin(φ + π + π * 0.25))

    # Ankle pitch: push-off just before toe-off, dorsiflexion during swing.
    l_ankle = ANKLE_AMP * math.sin(φ + π * 0.55)
    r_ankle = ANKLE_AMP * math.sin(φ + π + π * 0.55)

    # Lateral hip roll: small counter-swing to keep COM over stance leg.
    l_roll = -HIP_ROLL * math.cos(φ)
    r_roll =  HIP_ROLL * math.cos(φ + π)

    # ── arm parameters ───────────────────────────────────────────────
    ARM_AMP    = 0.30   # shoulder pitch swing (arm swings opposite to same-side leg)
    ELBOW_BEND = 1.28   # fixed elbow bend (natural carry angle)
    SHLD_ROLL  = 0.20   # fixed abduction

    l_shoulder = ARM_AMP * math.sin(φ + π)   # left arm in phase with right leg
    r_shoulder = ARM_AMP * math.sin(φ)        # right arm in phase with left leg

    return [
        # Left leg (indices 0-5)
        l_hip_pitch, l_roll, 0.,
        l_knee,
        l_ankle, 0.,
        # Right leg (indices 6-11)
        r_hip_pitch, r_roll, 0.,
        r_knee,
        r_ankle, 0.,
        # Waist (indices 12-14)
        0., 0., 0.,
        # Left arm (indices 15-21)
        l_shoulder, SHLD_ROLL, 0., ELBOW_BEND, 0., 0., 0.,
        # Right arm (indices 22-28)
        r_shoulder, -SHLD_ROLL, 0., ELBOW_BEND, 0., 0., 0.,
    ]


def _spot_walk_joints(phase: float) -> list[float]:
    """Spot trot gait joint angles at the given phase (0 → 2π = one trot cycle).

    Joint order (12 values after the 7-DOF free joint):
      0 fl_hx  1 fl_hy  2 fl_kn    (front-left)
      3 fr_hx  4 fr_hy  5 fr_kn    (front-right)
      6 hl_hx  7 hl_hy  8 hl_kn    (hind-left)
      9 hr_hx 10 hr_hy 11 hr_kn    (hind-right)

    Trot diagonal pairs: FL+HR (phase φ) and FR+HL (phase φ+π).
    """
    φ = phase
    π = math.pi

    HY_BASE  = 1.04    # standing hip flexion (rad)
    KN_BASE  = -1.8    # standing knee angle (rad, negative = knee folds rearward)
    HY_AMP   = 0.30    # hip pitch swing amplitude
    KN_LIFT  = 0.35    # extra knee bend during swing (foot lifts)
    HX_AMP   = 0.04    # small lateral (abduction) sway

    def _leg(φl: float) -> list[float]:
        hy = HY_BASE + HY_AMP * math.sin(φl)
        # Knee bends during the positive (swing) half and straightens in stance.
        kn = KN_BASE - KN_LIFT * max(0.0, math.sin(φl + π * 0.25))
        hx = HX_AMP * math.sin(φl)
        return [hx, hy, kn]

    # FL + HR diagonal share phase φ; FR + HL share φ+π.
    fl = _leg(φ)
    fr = _leg(φ + π)
    hl = _leg(φ + π)
    hr = _leg(φ)

    return fl + fr + hl + hr

ROBOT_TYPES: dict[str, dict[str, Any]] = {
    "g1": {
        # xml is relative to any scene dir that contains unitree_g1/
        # Fallback: use living-room-v1's copy
        "xml_candidates": [
            "unitree_g1/g1.xml",                        # scene-local copy
            "../living-room-v1/unitree_g1/g1.xml",      # from another scene
        ],
        "base_joint": "floating_base_joint",
        "standing_joints": _G1_STAND,
    },
    "spot": {
        "xml_candidates": ["boston_dynamics_spot/spot.xml"],
        "base_joint": "freejoint",
        "standing_joints": _SPOT_STAND,
    },
}


def _resolve_robot_xml(scene_dir: Path, robot_type: str) -> str:
    cfg = ROBOT_TYPES[robot_type]
    for candidate in cfg["xml_candidates"]:
        p = (scene_dir / candidate).resolve()
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Cannot find robot XML for type '{robot_type}' in scene dir {scene_dir}"
    )


# ---------------------------------------------------------------------------
# Per-scene config (camera, floor, default layout)
# ---------------------------------------------------------------------------
_SCENE_CONFIG: dict[str, dict[str, Any]] = {
    "living-room-v1": {
        # Floor top surface at z=0.2; G1 foot→root measured as 0.81 m → 1.01 m.
        "agent_base_heights": {"g1": 1.01},
        "has_bundled_robot": True,  # scene.xml already includes one G1
        "default_layout": ["g1", "g1"],
        "camera": {"distance": 7.0, "elevation": -55.0, "azimuth": 135.0,
                   "lookat": [0.0, 0.2, 0.4]},
    },
    "construction-v1": {
        # Empirically calibrated: construction site floor ≈ z=1.19.
        # G1: floor_z(1.19) + standing_height(0.79) = 1.98.
        # Spot: floor_z(1.19) + standing_height(0.46) = 1.65.
        "agent_base_heights": {"spot": 1.65, "g1": 1.98},
        "has_bundled_robot": False,
        "default_layout": ["g1", "g1", "g1", "g1", "spot", "spot"],
        "camera": {"distance": 20.0, "elevation": -45.0, "azimuth": 135.0,
                   "lookat": [0.0, 0.0, 1.2]},
    },
}

_FLOOR_KEYWORDS = ("terrain", "foundation", "floor", "slab", "grass", "sand")
_PREFIXES = ["A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_"]

_CACHE: dict[tuple, "SceneRenderer"] = {}


def get_renderer(scene_id: str,
                 agent_types: list[str] | None = None) -> "SceneRenderer":
    cfg = _SCENE_CONFIG.get(scene_id, {})
    if agent_types is None:
        agent_types = cfg.get("default_layout", ["g1"])
    key = (scene_id, tuple(agent_types))
    if key not in _CACHE:
        _CACHE[key] = SceneRenderer(scene_id, agent_types)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class SceneRenderer:
    def __init__(self, scene_id: str, agent_types: list[str],
                 movable_blocks: list[dict] | None = None) -> None:
        """movable_blocks: optional free-body box markers to inject into the scene
        before compile, so callers can move them via `_free_adr(name)`. Each entry:
            {"name": str, "half": float, "rgba": [r, g, b, a]}
        Needed for scenes (e.g. construction-v1) whose XML has no movable block
        bodies — without this, objects passed to render()/pose() are silently
        skipped because no free joint exists for them.
        """
        scene_dir = SCENES_DIR / scene_id
        scene_xml = str((scene_dir / "scene.xml").resolve())
        cfg = _SCENE_CONFIG.get(scene_id, {})

        base_heights_map: dict[str, float] = cfg.get("agent_base_heights", {})
        has_bundled = cfg.get("has_bundled_robot", False)
        self._camera_cfg: dict[str, Any] = cfg.get(
            "camera", {"distance": 10.0, "elevation": -45.0,
                       "azimuth": 135.0, "lookat": [0., 0., 1.]})

        spec = mujoco.MjSpec.from_file(scene_xml)

        # Record the number of lights in the scene BEFORE attaching robots so we
        # can zero-out the extra robot lights after compilation (robots embed their
        # own light sources which would otherwise massively over-expose the scene).
        _n_scene_lights = spec.compile().nlight

        # Per-agent state
        self._agent_types = list(agent_types)
        self._base_heights: list[float] = []
        self._standing_joints: list[list[float]] = []
        self._base_joint_names: list[str] = []

        # Attach robots: if scene has bundled robot it already provides agent[0]
        prefix_idx = 0
        for i, rtype in enumerate(agent_types):
            rtcfg = ROBOT_TYPES[rtype]
            self._base_heights.append(base_heights_map.get(rtype, 1.0))
            self._standing_joints.append(list(rtcfg["standing_joints"]))
            base_joint = rtcfg["base_joint"]

            if i == 0 and has_bundled:
                # Use the robot already in scene.xml unchanged (no prefix)
                self._base_joint_names.append(base_joint)
            else:
                prefix = _PREFIXES[prefix_idx]
                prefix_idx += 1
                child = mujoco.MjSpec.from_file(
                    _resolve_robot_xml(scene_dir, rtype))
                spec.attach(child, prefix=prefix,
                            frame=spec.worldbody.add_frame())
                self._base_joint_names.append(f"{prefix}{base_joint}")

        # Inject movable block markers (free-body boxes) so they can be posed.
        for blk in (movable_blocks or []):
            half = float(blk.get("half", 0.35))
            body = spec.worldbody.add_body(name=blk["name"], pos=[0.0, 0.0, half])
            body.add_freejoint()
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [half, half, half]
            geom.rgba = blk.get("rgba", [0.85, 0.35, 0.12, 1.0])
            geom.contype = 0
            geom.conaffinity = 0

        self.model = spec.compile()
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 960)

        # Lighting corrections for PBR-texture scenes.
        # Zero out any lights added by robot models — each G1 adds 1 and each Spot
        # adds 2 extra directional lights that wash out the entire scene.
        self.model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
        self.model.vis.headlight.diffuse[:] = [0.9, 0.9, 0.9]
        for li in range(self.model.nlight):
            if li >= _n_scene_lights:
                # Robot-added light — disable it entirely
                self.model.light_diffuse[li, :] = 0.0
                self.model.light_ambient[li, :] = 0.0
            else:
                self.model.light_ambient[li] = np.clip(
                    self.model.light_ambient[li] * 2.0, 0, 1)
                self.model.light_diffuse[li] = np.clip(
                    self.model.light_diffuse[li] * 2.0, 0, 1)
        # Override floor/terrain geom colours to match the sandy construction site ground.
        # MuJoCo's Phong renderer doesn't project 2D textures onto mesh geoms without
        # UV coords, so we use a solid rgba derived from the site's panorama ground sample.
        # Sand/dirt colour: mean of the panorama's bottom ground strip (~138, 116, 93).
        _SAND_RGBA = [0.58, 0.48, 0.39, 1.0]
        floor_matids: set[int] = set()
        for gi in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gi]
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(kw in bname.lower() for kw in _FLOOR_KEYWORDS):
                mid = int(self.model.geom_matid[gi])
                if mid >= 0:
                    floor_matids.add(mid)
                self.model.geom_rgba[gi] = _SAND_RGBA
        for mid in floor_matids:
            self.model.mat_texid[mid] = -1
            self.model.mat_rgba[mid] = _SAND_RGBA
            self.model.mat_emission[mid] = 0.0
            self.model.mat_reflectance[mid] = 0.03
        for i in range(self.model.nmat):
            if i in floor_matids:
                continue
            rgba = self.model.mat_rgba[i]
            rgba[3] = max(rgba[3], 1.0)
            rgba[:3] = np.where(rgba[:3] < 0.08, 0.22, rgba[:3])

        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self._base_adr = [self._jadr(n) for n in self._base_joint_names]
        self._free_adr_cache: dict[str, int] = {}

        # Identify which geoms belong to robots vs. static scene so callers
        # can use fast_render=True to hide the expensive scene geoms.
        _ROBOT_KW = ("hip", "knee", "leg", "arm", "shoulder", "elbow", "wrist",
                     "hand", "trunk", "torso", "pelvis", "foot", "ankle", "head",
                     "logo", "fl_", "fr_", "hl_", "hr_", "link", "waist",
                     "contour", "rubber")
        self._scene_geom_ids: list[int] = []
        for gi in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gi]
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not any(kw in bname.lower() for kw in _ROBOT_KW):
                self._scene_geom_ids.append(gi)

    # ------------------------------------------------------------------
    def _jadr(self, name: str) -> int:
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[j]) if j >= 0 else -1

    def _free_adr(self, body: str) -> int:
        if body not in self._free_adr_cache:
            adr = -1
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if b >= 0:
                for j in range(self.model.njnt):
                    if (self.model.jnt_bodyid[j] == b and
                            self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                        adr = int(self.model.jnt_qposadr[j])
                        break
            self._free_adr_cache[body] = adr
        return self._free_adr_cache[body]

    # ------------------------------------------------------------------
    def pose(self, agents: list[tuple[float, float, float]],
             objects: dict[str, tuple[float, float, float, float]],
             phases: list[float] | None = None) -> None:
        """Set qpos for all agents and objects (does NOT call mj_forward).

        phases: per-agent gait phase (0..2π). When provided the robot is posed
                in its walking stance; when None it uses the static standing pose.
        """
        from sim.agent_spacing import spread_all
        if len(agents) > 1:
            ad = {str(i): {"x": a[0], "y": a[1]}
                  for i, a in enumerate(agents)}
            spread_all(ad)
            agents = [(ad[str(i)]["x"], ad[str(i)]["y"], a[2])
                      for i, a in enumerate(agents)]

        for i, ((x, y, yaw), adr) in enumerate(zip(agents, self._base_adr)):
            if adr < 0:
                continue
            bh = self._base_heights[i]
            rtype = self._agent_types[i]

            # Choose joint angles: walking pose when phase supplied, else standing.
            if phases is not None:
                φ = float(phases[i]) if i < len(phases) else 0.0
                if rtype == "g1":
                    sj = _g1_walk_joints(φ)
                elif rtype == "spot":
                    sj = _spot_walk_joints(φ)
                else:
                    sj = self._standing_joints[i]
            else:
                sj = self._standing_joints[i]

            self.data.qpos[adr:adr + 7] = [
                x, y, bh, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
            if sj:
                self.data.qpos[adr + 7:adr + 7 + len(sj)] = sj

        for body, (x, y, z, yaw) in objects.items():
            a = self._free_adr(body)
            if a >= 0:
                self.data.qpos[a:a + 7] = [
                    x, y, z, math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]

    def _forward(self) -> None:
        prev = self.model.opt.disableflags
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        mujoco.mj_forward(self.model, self.data)
        self.model.opt.disableflags = prev

    def render(self, agents: list[tuple[float, float, float]],
               objects: dict[str, tuple[float, float, float, float]],
               width: int = 960, height: int = 640,
               camera_override: dict | None = None,
               fast: bool = False,
               phases: list[float] | None = None) -> bytes:
        """agents: [(x,y,yaw), ...]; objects: {body:(x,y,z,yaw)}. Returns PNG.

        phases: per-agent gait phase (0..2π). Pass to animate walking.
                Advance by WALK_ANGULAR_VEL each frame for smooth animation.
        fast=True: hide static scene geoms (only robots shown) — ~7x faster for video.
        """
        self.pose(agents, objects, phases=phases)
        self._forward()

        cc = camera_override or self._camera_cfg
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = cc["distance"]
        cam.elevation = cc["elevation"]
        cam.azimuth = cc["azimuth"]
        cam.lookat[:] = cc["lookat"]

        scene_option: mujoco.MjvOption | None = None
        if fast:
            for gi in self._scene_geom_ids:
                self.model.geom_group[gi] = 3
            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        else:
            for gi in self._scene_geom_ids:
                self.model.geom_group[gi] = 0

        r = mujoco.Renderer(self.model, height=height, width=width)
        try:
            if scene_option is not None:
                r.update_scene(self.data, cam, scene_option=scene_option)
            else:
                r.update_scene(self.data, cam)
            px = r.render()
        finally:
            r.close()

        import io
        buf = io.BytesIO()
        Image.fromarray(px).save(buf, format="PNG")
        return buf.getvalue()
