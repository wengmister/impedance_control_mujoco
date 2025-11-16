"""
Joint-space bilateral teleoperation demo for the two-arm scene.

xarm master franka follower
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from util.torque_viz import TorqueIndicator
from util.traj_viz import TrajectoryTrail


# MODEL_XML = Path("bilateral_impedance") / "scene_bilateral_duoxarm.xml"
MODEL_XML = Path("bilateral_impedance") / "scene_bilateral_xarm_franka.xml"

# Home pose pulled from the single-arm controller demo.
HOME_QPOS_XARM = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])
HOME_QPOS_FRANKA = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

# Coupling gains between corresponding joints (Nm/rad, Nms/rad).
K_COUPLE = np.array([200, 200, 200, 200, 100, 100, 100])
D_COUPLE = np.array([30, 30, 25, 20, 12, 8, 6])

# Mapping between Franka joint angles and the xArm reference.
FOLLOWER_SCALE = np.array([1, 1, 1, 1, 1, 1, 1])
FOLLOWER_OFFSET = np.array([0.0, 0.0, 0.0, 3.14, 0.0, -0.157, 0.0])

@dataclass
class ArmHandles:
    qpos_idx: np.ndarray
    qvel_idx: np.ndarray
    act_idx: np.ndarray


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return C(q, qd) + g(q) using Recursive Newton-Euler."""
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def make_xarm_handles(model: mujoco.MjModel, prefix: str, dof: int) -> ArmHandles:
    """Collect qpos/qvel/actuator indices for an attached arm."""
    qpos_idx = []
    qvel_idx = []
    act_idx = []
    for joint_id in range(1, dof + 1):
        joint_name = f"{prefix}joint{joint_id}"
        actuator_name = f"{prefix}act{joint_id}"
        mj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        mj_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        qpos_idx.append(model.jnt_qposadr[mj_joint_id])
        qvel_idx.append(model.jnt_dofadr[mj_joint_id])
        act_idx.append(mj_act_id)
        # print(f"{joint_name}: qpos {model.jnt_qposadr[mj_joint_id]}, qvel {model.jnt_dofadr[mj_joint_id]}, act {mj_act_id}")

    return ArmHandles(
        qpos_idx=np.asarray(qpos_idx, dtype=int),
        qvel_idx=np.asarray(qvel_idx, dtype=int),
        act_idx=np.asarray(act_idx, dtype=int),
    )

def make_franka_handles(model: mujoco.MjModel, prefix: str, dof: int) -> ArmHandles:
    """Collect qpos/qvel/actuator indices for an attached arm."""
    qpos_idx = []
    qvel_idx = []
    act_idx = []
    for joint_id in range(1, dof + 1):
        joint_name = f"{prefix}fr3_joint{joint_id}"
        actuator_name = f"{prefix}fr3_torque{joint_id}"
        mj_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        mj_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        qpos_idx.append(model.jnt_qposadr[mj_joint_id])
        qvel_idx.append(model.jnt_dofadr[mj_joint_id])
        act_idx.append(mj_act_id)
        # print(f"{joint_name}: qpos {model.jnt_qposadr[mj_joint_id]}, qvel {model.jnt_dofadr[mj_joint_id]}, act {mj_act_id}")

    return ArmHandles(
        qpos_idx=np.asarray(qpos_idx, dtype=int),
        qvel_idx=np.asarray(qvel_idx, dtype=int),
        act_idx=np.asarray(act_idx, dtype=int),
    )


def set_home_configuration(data: mujoco.MjData, handles: ArmHandles, home: Sequence[float]) -> None:
    """Place an arm at the provided joint configuration."""
    data.qpos[handles.qpos_idx] = home
    data.qvel[handles.qvel_idx] = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bilateral impedance teleoperation demo")
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Samples to keep in each end-effector trail (0 disables)",
    )
    parser.add_argument(
        "--trail-stride",
        type=int,
        default=25,
        help="Simulation steps between trail samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    master = make_xarm_handles(model, prefix="master_", dof=HOME_QPOS_XARM.size)
    follower = make_franka_handles(model, prefix="follower_", dof=HOME_QPOS_FRANKA.size)

    # Start both arms from the same comfortable pose.
    set_home_configuration(data, master, HOME_QPOS_XARM)
    set_home_configuration(data, follower, HOME_QPOS_FRANKA)
    mujoco.mj_forward(model, data)

    print("Launching bilateral teleoperation demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  joints per arm: {HOME_QPOS_XARM.size}")
    print("  master prefix: master_")
    print("  follower prefix: follower_")

    trail_master = (
        TrajectoryTrail(maxlen=args.trail_length, stride=args.trail_stride)
        if args.trail_length > 0
        else None
    )
    trail_follower = (
        TrajectoryTrail(maxlen=args.trail_length, stride=args.trail_stride)
        if args.trail_length > 0
        else None
    )
    master_site = None
    follower_site = None
    if args.trail_length > 0:
        master_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "master_attachment_site")
        follower_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "follower_attachment_site")
        if master_site < 0 or follower_site < 0:
            raise ValueError("Attachment sites not found for master or follower arms.")

    torque_indicator = TorqueIndicator(scale=0.12)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias = compute_bias_forces(model, data)

            q_master = data.qpos[master.qpos_idx]
            qd_master = data.qvel[master.qvel_idx]
            bias_master = bias[master.qvel_idx]

            q_follower = data.qpos[follower.qpos_idx]
            qd_follower = data.qvel[follower.qvel_idx]
            bias_follower = bias[follower.qvel_idx]
            q_follower_virtual = (q_follower + FOLLOWER_OFFSET) * FOLLOWER_SCALE
            qd_follower_virtual = qd_follower * FOLLOWER_SCALE

            # Virtual spring-damper coupling between matched joints.
            coupling = K_COUPLE * (q_master - q_follower_virtual) + D_COUPLE * (
                qd_master - qd_follower_virtual
            )

            tau_master = bias_master - coupling
            tau_follower = bias_follower + coupling

            data.ctrl[master.act_idx] = tau_master
            data.ctrl[follower.act_idx] = tau_follower

            mujoco.mj_step(model, data)
            ngeom = torque_indicator.draw(viewer, model, data, base_index=0)
            if trail_master is not None:
                trail_master.update(data.site_xpos[master_site])
                ngeom = trail_master.draw(viewer, base_index=ngeom)
            if trail_follower is not None:
                trail_follower.update(data.site_xpos[follower_site])
                ngeom = trail_follower.draw(viewer, base_index=ngeom)
            viewer.user_scn.ngeom = ngeom
            viewer.sync()


if __name__ == "__main__":
    main()
