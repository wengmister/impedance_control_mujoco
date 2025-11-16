"""
Virtual-proxy bilateral teleoperation demo.

Implements the "virtual proxy / virtual coupling" architecture from
BILATERAL_TELEOP.md using the xArm master and Franka follower scene.

The proxy lives in task space (EE position only) and is coupled to both
arms with independent spring-damper pairs. The follower tracks the proxy
through task-space impedance, while the master feels both the proxy
coupling force and any reflected environment wrench measured at the
follower end effector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from util.traj_viz import TrajectoryTrail


MODEL_XML = Path("bilateral_impedance") / "scene_bilateral_xarm_franka.xml"

# Reference poses taken from the single-arm demos.
HOME_QPOS_XARM = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])
HOME_QPOS_FRANKA = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Virtual coupling gains (Cartesian translation/rotation).
K_MASTER_POS = np.diag([600.0, 600.0, 600.0])
D_MASTER_POS = np.diag([60.0, 60.0, 60.0])
K_MASTER_ROT = np.diag([45.0, 45.0, 45.0])
D_MASTER_ROT = np.diag([5.0, 5.0, 5.0])

K_FOLLOWER_POS = np.diag([900.0, 900.0, 1200.0])
D_FOLLOWER_POS = np.diag([80.0, 80.0, 120.0])
K_FOLLOWER_ROT = np.diag([60.0, 60.0, 80.0])
D_FOLLOWER_ROT = np.diag([6.0, 6.0, 8.0])

# Proxy inertial/damping properties.
PROXY_MASS = 2.0
PROXY_DAMP = 40.0
PROXY_INERTIA = 0.05
PROXY_ANG_DAMP = 2.0

# Desired offset (world frame) from the master proxy to the follower EE target.
FOLLOWER_TARGET_OFFSET = np.array([-0.7, 0.0, 0.0])
FOLLOWER_TARGET_ORIENT_OFFSET = np.array([1.0, 0.0, 0.0, 0.0])

# Scale applied when reflecting environment force back to the master.
FORCE_REFLECTION_SCALE = 1.0

# Names of the important sites/bodies after prefixing.
MASTER_SITE = "master_attachment_site"
FOLLOWER_SITE = "follower_attachment_site"
FOLLOWER_EE_BODY = "follower_fr3_link7"


@dataclass
class ArmHandles:
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray
    act_idx: np.ndarray


@dataclass
class VirtualProxy:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    angvel: np.ndarray

    def integrate(self, lin_acc: np.ndarray, ang_acc: np.ndarray, dt: float) -> None:
        self.vel += lin_acc * dt
        self.pos += self.vel * dt
        self.angvel += ang_acc * dt
        omega_quat = np.array([0.0, *self.angvel])
        q_dot = 0.5 * quat_multiply(omega_quat, self.quat)
        self.quat += q_dot * dt
        self.quat = quat_normalize(self.quat)


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return C(q, qd) + g(q) using Recursive Newton-Euler."""
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def make_arm_handles(
    model: mujoco.MjModel, prefix: str, joint_names: Sequence[str], actuator_names: Sequence[str]
) -> ArmHandles:
    """Resolve joint/actuator indices for a prefixed arm."""
    if len(joint_names) != len(actuator_names):
        raise ValueError("Joint and actuator lists must match in length.")
    qpos_idx = []
    qvel_idx = []
    act_idx = []
    for j_name, a_name in zip(joint_names, actuator_names):
        mj_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{j_name}")
        mj_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{a_name}")
        qpos_idx.append(model.jnt_qposadr[mj_joint])
        qvel_idx.append(model.jnt_dofadr[mj_joint])
        act_idx.append(mj_act)
    return ArmHandles(
        qpos_idx=np.asarray(qpos_idx, dtype=int),
        qvel_idx=np.asarray(qvel_idx, dtype=int),
        act_idx=np.asarray(act_idx, dtype=int),
    )


def set_home_configuration(data: mujoco.MjData, handles: ArmHandles, home: Sequence[float]) -> None:
    """Place an arm at the provided joint configuration."""
    data.qpos[handles.qpos_idx] = home
    data.qvel[handles.qvel_idx] = 0.0


def mat_to_quat(xmat: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo xmat (9 elements) to a normalized quaternion [w, x, y, z]."""
    m = xmat.reshape(3, 3)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / (np.linalg.norm(quat) + 1e-12)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-12)


def quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return a small-angle rotation vector to align current orientation with target."""
    delta = quat_multiply(target, quat_conjugate(current))
    if delta[0] < 0.0:
        delta = -delta
    return 2.0 * delta[1:]


def site_kinematics(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return site pose, kinematics, and Jacobians."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    pos = data.site_xpos[site_id].copy()
    vel = jacp @ data.qvel
    angvel = jacr @ data.qvel
    quat = mat_to_quat(data.site_xmat[site_id])
    return pos, vel, quat, angvel, jacp, jacr


def external_force_on_body(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """World-space external force acting on a body (first 3 components)."""
    return data.cfrc_ext[body_id, :3].copy()


def external_torque_on_body(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """World-space external torque acting on a body (last 3 components)."""
    return data.cfrc_ext[body_id, 3:].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual proxy bilateral teleoperation demo")
    parser.add_argument(
        "--trail-length",
        type=int,
        default=100,
        help="Samples to keep in the EE trails (0 disables)",
    )
    parser.add_argument(
        "--trail-stride",
        type=int,
        default=50,
        help="Simulation steps between trail samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    master = make_arm_handles(
        model,
        prefix="master_",
        joint_names=[f"joint{i}" for i in range(1, 8)],
        actuator_names=[f"act{i}" for i in range(1, 8)],
    )
    follower = make_arm_handles(
        model,
        prefix="follower_",
        joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
        actuator_names=[f"fr3_torque{i}" for i in range(1, 8)],
    )

    set_home_configuration(data, master, HOME_QPOS_XARM)
    set_home_configuration(data, follower, HOME_QPOS_FRANKA)
    mujoco.mj_forward(model, data)

    master_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, MASTER_SITE)
    follower_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOLLOWER_SITE)
    follower_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOLLOWER_EE_BODY)
    if master_site < 0 or follower_site < 0 or follower_body < 0:
        raise ValueError("Failed to resolve master/follower site or body IDs.")

    master_pos = data.site_xpos[master_site].copy()
    master_quat = mat_to_quat(data.site_xmat[master_site])
    proxy = VirtualProxy(
        pos=master_pos.copy(),
        vel=np.zeros(3),
        quat=master_quat.copy(),
        angvel=np.zeros(3),
    )

    print("Launching virtual proxy teleoperation demo...")
    print(f"  model: {MODEL_XML}")
    print("  master arm: xArm7 (prefix master_)")
    print("  follower arm: Franka (prefix follower_)")

    trail_master = (
        TrajectoryTrail(maxlen=args.trail_length, stride=args.trail_stride, color=np.array([0.2, 0.8, 0.2, 0.7]))
        if args.trail_length > 0
        else None
    )
    trail_follower = (
        TrajectoryTrail(maxlen=args.trail_length, stride=args.trail_stride, color=np.array([0.8, 0.2, 0.2, 0.7]))
        if args.trail_length > 0
        else None
    )
    trail_proxy = (
        TrajectoryTrail(maxlen=args.trail_length, stride=args.trail_stride, color=np.array([0.2, 0.3, 0.9, 0.8]))
        if args.trail_length > 0
        else None
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias = compute_bias_forces(model, data)

            (
                master_pos,
                master_vel,
                master_quat,
                master_angvel,
                jac_master_p,
                jac_master_r,
            ) = site_kinematics(model, data, master_site)
            (
                follower_pos,
                follower_vel,
                follower_quat,
                follower_angvel,
                jac_follower_p,
                jac_follower_r,
            ) = site_kinematics(model, data, follower_site)

            follower_target_pos = proxy.pos + FOLLOWER_TARGET_OFFSET
            follower_target_quat = quat_multiply(proxy.quat, FOLLOWER_TARGET_ORIENT_OFFSET)

            force_master = K_MASTER_POS @ (proxy.pos - master_pos) + D_MASTER_POS @ (proxy.vel - master_vel)
            torque_master = K_MASTER_ROT @ quat_error(proxy.quat, master_quat) + D_MASTER_ROT @ (
                proxy.angvel - master_angvel
            )

            force_follower = K_FOLLOWER_POS @ (follower_target_pos - follower_pos) + D_FOLLOWER_POS @ (
                proxy.vel - follower_vel
            )
            torque_follower = K_FOLLOWER_ROT @ quat_error(follower_target_quat, follower_quat) + D_FOLLOWER_ROT @ (
                proxy.angvel - follower_angvel
            )

            env_force = external_force_on_body(data, follower_body) * FORCE_REFLECTION_SCALE
            env_torque = external_torque_on_body(data, follower_body) * FORCE_REFLECTION_SCALE

            proxy_lin_acc = (-force_master - force_follower + env_force - PROXY_DAMP * proxy.vel) / PROXY_MASS
            proxy_ang_acc = (-torque_master - torque_follower + env_torque - PROXY_ANG_DAMP * proxy.angvel) / PROXY_INERTIA
            proxy.integrate(proxy_lin_acc, proxy_ang_acc, model.opt.timestep)

            tau_master = (
                jac_master_p[:, master.qvel_idx].T @ (force_master + env_force)
                + jac_master_r[:, master.qvel_idx].T @ (torque_master + env_torque)
                + bias[master.qvel_idx]
            )
            tau_follower = (
                jac_follower_p[:, follower.qvel_idx].T @ force_follower
                + jac_follower_r[:, follower.qvel_idx].T @ torque_follower
                + bias[follower.qvel_idx]
            )

            data.ctrl[master.act_idx] = tau_master
            data.ctrl[follower.act_idx] = tau_follower

            mujoco.mj_step(model, data)

            ngeom = 0
            if trail_master is not None:
                trail_master.update(master_pos)
                ngeom = trail_master.draw(viewer, base_index=ngeom)
            if trail_follower is not None:
                trail_follower.update(follower_pos)
                ngeom = trail_follower.draw(viewer, base_index=ngeom)
            if trail_proxy is not None:
                trail_proxy.update(proxy.pos)
                ngeom = trail_proxy.draw(viewer, base_index=ngeom)
            viewer.user_scn.ngeom = ngeom
            viewer.sync()


if __name__ == "__main__":
    main()
