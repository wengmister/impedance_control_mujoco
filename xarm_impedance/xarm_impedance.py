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
SINE_FREQUENCY_HZ = 0.1

# End-effector sweep definition (world-Y translation).
EE_SITE = "attachment_site"
EE_AXIS = np.array([0.0, 1.0, 0.0])
EE_SWEEP_AMPLITUDE = 0.05  # meters
EE_SWEEP_FREQUENCY_HZ = 0.05
KX = np.array([1000.0, 1000.0, 1000.0])
DX = np.array([50.0, 50.0, 50.0])

# IK solver parameters.
IK_MAX_ITERS = 30
IK_TOL = 1e-4
IK_DAMPING = 1e-3


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


def ee_reference(t: float, home_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Desired EE trajectory: sinusoid along the specified axis."""
    omega = 2.0 * np.pi * EE_SWEEP_FREQUENCY_HZ
    x_des = home_pos + EE_AXIS * EE_SWEEP_AMPLITUDE * np.sin(omega * t)
    xd_des = EE_AXIS * EE_SWEEP_AMPLITUDE * omega * np.cos(omega * t)
    return x_des, xd_des


def task_space_impedance_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bias: np.ndarray,
    site_id: int,
    x_des: np.ndarray,
    xd_des: np.ndarray,
) -> np.ndarray:
    """Compute tau = J^T F + bias for a task-space impedance."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    x = data.site_xpos[site_id].copy()
    xd = jacp @ data.qvel
    force = -KX * (x - x_des) - DX * (xd - xd_des)
    tau = jacp.T @ force + bias
    return tau


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
) -> np.ndarray:
    """Levenberg-Marquardt IK for site position only."""
    q = q_init.copy()
    for _ in range(IK_MAX_ITERS):
        data.qpos[: model.nq] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < IK_TOL:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        JJ = jacp @ jacp.T + IK_DAMPING * np.eye(3)
        dq = jacp.T @ np.linalg.solve(JJ, err)
        q += dq
    return q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xArm7 controller demos")
    parser.add_argument(
        "--mode",
        choices=("gravity", "joint_sine", "ee_sine"),
        default="ee_sine",
        help="Controller mode to run",
    )
    parser.add_argument(
        "--space",
        choices=("joint", "task"),
        default="joint",
        help="Impedance space for ee_sine mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Start from the home posture to avoid large initial errors.
    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[: model.nv] = 0.0
    mujoco.mj_forward(model, data)

    ee_site_id = None
    ee_home_pos = None
    ik_workspace = None
    prev_q_des = None
    prev_t = None
    if args.mode == "ee_sine":
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if ee_site_id < 0:
            raise ValueError(f"Site '{EE_SITE}' not found in model.")
        ee_home_pos = data.site_xpos[ee_site_id].copy()
        if args.space == "joint":
            ik_workspace = mujoco.MjData(model)
            prev_q_des = HOME_QPOS.copy()
            prev_t = data.time

    print("Launching xArm7 controller demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  joints: {model.njnt}, actuators: {model.nu}")
    print(f"  mode: {args.mode}")
    if args.mode == "ee_sine":
        print(f"  space: {args.space}")

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
            elif args.mode == "ee_sine":
                x_des, xd_des = ee_reference(data.time, ee_home_pos)
                if args.space == "task":
                    tau = task_space_impedance_torque(model, data, bias, ee_site_id, x_des, xd_des)
                elif args.space == "joint":
                    if ik_workspace is None:
                        raise RuntimeError("IK workspace not initialized.")
                    q_guess = prev_q_des if prev_q_des is not None else q
                    q_des = solve_position_ik(model, ik_workspace, ee_site_id, x_des, q_guess)
                    last_t = prev_t if prev_t is not None else data.time
                    dt = max(data.time - last_t, 1e-6)
                    if prev_q_des is None:
                        qd_des = np.zeros_like(q_des)
                    else:
                        qd_des = (q_des - prev_q_des) / dt
                    tau = impedance_torque(q, qd, q_des, qd_des, KP, KD, bias)
                    prev_q_des = q_des
                    prev_t = data.time
                else:
                    raise ValueError(f"Unsupported space '{args.space}'")
            else:
                raise ValueError(f"Unknown mode '{args.mode}'")
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
