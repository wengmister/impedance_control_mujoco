"""
Joint-space impedance controller demo for the passive xArm7 MuJoCo model.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import mujoco
import mujoco.viewer


MODEL_DIR = Path("mujoco_menagerie/ufactory_xarm7")
MODEL_XML = MODEL_DIR / "scene_passive.xml"

# Keyframe pose from the MJCF is a convenient home configuration.
HOME_QPOS = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])
HOME_QVEL = np.zeros_like(HOME_QPOS)

# Joint-space gains (Nm/rad and Nms/rad). 
# Tuned for a stiff, well-damped feel.
KP = np.array([300, 300, 250, 200, 150, 120, 100])
KD = np.array([40, 40, 35, 25, 18, 14, 10])


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return C(q, qd) + g(q) using Recursive Newton-Euler."""
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def impedance_torque(
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Joint-space impedance law with gravity/bias compensation."""
    return -kp * (q - q_des) - kd * (qd - qd_des) + bias


def main():
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Start from the home posture to avoid large initial errors.
    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[: model.nv] = HOME_QVEL
    mujoco.mj_forward(model, data)

    print("Launching xArm7 impedance controller demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  joints: {model.njnt}, actuators: {model.nu}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            q = data.qpos[: model.nq].copy()
            qd = data.qvel[: model.nv].copy()
            bias = compute_bias_forces(model, data)
            tau = impedance_torque(q, qd, HOME_QPOS, HOME_QVEL, KP, KD, bias)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
