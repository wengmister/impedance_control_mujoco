"""
Joint-space bilateral teleoperation demo for the two-arm scene.

The master arm is torque-controlled and coupled to the follower arm
through a virtual spring-damper on their joint differences. Any torque
disturbance on the follower is reflected back to the master, enabling
simple haptic feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np


MODEL_XML = Path("bilateral_impedance") / "scene_bilateral.xml"

# Home pose pulled from the single-arm controller demo.
HOME_QPOS = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])

# Coupling gains between corresponding joints (Nm/rad, Nms/rad).
K_COUPLE = np.array([50, 50, 50, 50, 15, 10, 10])
D_COUPLE = np.array([30, 30, 25, 20, 12, 8, 6])


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


def make_arm_handles(model: mujoco.MjModel, prefix: str, dof: int) -> ArmHandles:
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


def set_home_configuration(data: mujoco.MjData, handles: ArmHandles, home: Sequence[float]) -> None:
    """Place an arm at the provided joint configuration."""
    data.qpos[handles.qpos_idx] = home
    data.qvel[handles.qvel_idx] = 0.0


def main() -> None:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    master = make_arm_handles(model, prefix="master_", dof=HOME_QPOS.size)
    follower = make_arm_handles(model, prefix="follower_", dof=HOME_QPOS.size)

    # Start both arms from the same comfortable pose.
    set_home_configuration(data, master, HOME_QPOS)
    set_home_configuration(data, follower, HOME_QPOS)
    mujoco.mj_forward(model, data)

    print("Launching bilateral teleoperation demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  joints per arm: {HOME_QPOS.size}")
    print("  master prefix: master_")
    print("  follower prefix: follower_")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias = compute_bias_forces(model, data)

            q_master = data.qpos[master.qpos_idx]
            qd_master = data.qvel[master.qvel_idx]
            bias_master = bias[master.qvel_idx]

            q_follower = data.qpos[follower.qpos_idx]
            qd_follower = data.qvel[follower.qvel_idx]
            bias_follower = bias[follower.qvel_idx]

            # Virtual spring-damper coupling between matched joints.
            coupling = K_COUPLE * (q_master - q_follower) + D_COUPLE * (qd_master - qd_follower)

            tau_master = bias_master - coupling
            tau_follower = bias_follower + coupling

            data.ctrl[master.act_idx] = tau_master
            data.ctrl[follower.act_idx] = tau_follower

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
