"""
Joint-space controller demos for the passive xArm7 MuJoCo model.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_DIR = Path("mujoco_menagerie/ufactory_xarm7")
MODEL_XML = MODEL_DIR / "scene_passive.xml"

# Keyframe pose from the MJCF is a convenient home configuration.
HOME_QPOS = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])
HOME_QVEL = np.zeros_like(HOME_QPOS)

# Joint-space gains (Nm/rad and Nms/rad). 
# Tuned for a stiff, well-damped feel.
KP = np.array([50, 50, 50, 50, 10, 10, 10])
KD = np.array([40, 40, 35, 25, 18, 14, 10])

# Small joint-space sine sweep around home pose.
SINE_AMPLITUDE = np.array([0.00, 0.0, 0.05, 0.05, 0.1, 0.1, 0.1])
SINE_FREQUENCY_HZ = 0.01


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


def joint_sine_reference(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple joint-space sine sweep around the home pose."""
    omega = 2.0 * np.pi * SINE_FREQUENCY_HZ
    q_des = HOME_QPOS + SINE_AMPLITUDE * np.sin(omega * t)
    qd_des = SINE_AMPLITUDE * omega * np.cos(omega * t)
    return q_des, qd_des


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xArm7 controller demos")
    parser.add_argument(
        "--mode",
        choices=("gravity", "joint_sine", "ee_traj"),
        default="joint_sine",
        help="Controller mode to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "ee_traj":
        print("End-effector trajectory mode not implemented yet.")
        return

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Start from the home posture to avoid large initial errors.
    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[: model.nv] = 0.0
    mujoco.mj_forward(model, data)

    print("Launching xArm7 controller demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  joints: {model.njnt}, actuators: {model.nu}")
    print(f"  mode: {args.mode}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            q = data.qpos[: model.nq].copy()
            qd = data.qvel[: model.nv].copy()
            bias = compute_bias_forces(model, data)
            if args.mode == "gravity":
                tau = bias
            elif args.mode == "joint_sine":
                q_des, qd_des = joint_sine_reference(data.time)
                tau = impedance_torque(q, qd, q_des, qd_des, KP, KD, bias)
            else:
                raise ValueError(f"Unknown mode '{args.mode}'")
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
