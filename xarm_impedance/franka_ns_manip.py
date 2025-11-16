"""
Null-space manipulation demo for xArm7: stiff task-space impedance + secondary joint objectives.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from util.torque_viz import TorqueIndicator
from util.traj_viz import TrajectoryTrail

MODEL_DIR = Path("xarm_impedance/scene")
MODEL_XML = MODEL_DIR / "scene_franka_passive.xml"
EE_SITE = "franka_attachment_site"

HOME_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.57079, -0.7853])

KX = np.array([3000.0, 2000.0, 2000.0])
DX = np.array([100.0, 100.0, 100.0])
KR = np.array([400.0, 400.0, 400.0])
DR = np.array([5.0, 5.0, 5.0])

K_NULL = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 1.0])
D_NULL = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2])


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def rotation_error_world(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    R_err = R_des.T @ R
    e_local = 0.5 * np.array(
        [R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]]
    )
    return R_des @ e_local


def task_space_wrench(model, data, site_id, x_des, xd_des, R_des, omega_des):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    x = data.site_xpos[site_id]
    xd = jacp @ data.qvel
    R = data.site_xmat[site_id].reshape(3, 3)
    e_R = rotation_error_world(R, R_des)
    omega = jacr @ data.qvel

    F = -KX * (x - x_des) - DX * (xd - xd_des)
    M = -KR * e_R - DR * (omega - omega_des)
    return jacp.T @ F + jacr.T @ M


def nullspace_projector(jac: np.ndarray) -> np.ndarray:
    # J shape (m,n), projector onto joint nullspace: I - J^T (JJ^T)^-1 J
    JJt = jac @ jac.T
    reg = 1e-6 * np.eye(JJt.shape[0])
    JJt_inv = np.linalg.pinv(JJt + reg)
    return np.eye(jac.shape[1]) - jac.T @ JJt_inv @ jac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xArm7 null-space manipulation demo")
    parser.add_argument("--mode", choices=("hold", "figure8"), default="figure8")
    parser.add_argument("--trail-length", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not MODEL_XML.exists():
        raise FileNotFoundError(MODEL_XML)

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    if site_id < 0:
        raise ValueError("EE site not found")
    home_pos = data.site_xpos[site_id].copy()
    home_rot = data.site_xmat[site_id].reshape(3, 3).copy()

    trail = TrajectoryTrail(maxlen=args.trail_length, stride=25) if args.trail_length > 0 else None
    torque_indicator = TorqueIndicator()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            bias = compute_bias_forces(model, data)

            if args.mode == "figure8":
                t = data.time
                theta = 2.0 * np.pi * 0.05 * t
                y = 0.2 * np.cos(theta)
                z = 0.2 * np.sin(theta) * np.cos(theta)
                dy = -0.2 * 2.0 * np.pi * 0.05 * np.sin(theta)
                dz = 0.2 * 2.0 * np.pi * 0.05 * np.cos(2 * theta)
                x_des = home_pos + np.array([0.0, y, z])
                xd_des = np.array([0.0, dy, dz])
            else:
                x_des = home_pos
                xd_des = np.zeros(3)

            R_des = home_rot
            omega_des = np.zeros(3)

            tau_task = task_space_wrench(model, data, site_id, x_des, xd_des, R_des, omega_des)

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])
            N = nullspace_projector(J)

            q_err = HOME_QPOS - data.qpos[: model.nq]
            qd = data.qvel[: model.nv]
            # tau_null = K_NULL * q_err - D_NULL * qd
            tau_null = -D_NULL * qd
            tau = tau_task + N @ tau_null + bias

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            ngeom = torque_indicator.draw(viewer, model, data, base_index=0)
            if trail is not None:
                trail.update(data.site_xpos[site_id])
                ngeom = trail.draw(viewer, base_index=ngeom)
            viewer.user_scn.ngeom = ngeom
            viewer.sync()


if __name__ == "__main__":
    main()
