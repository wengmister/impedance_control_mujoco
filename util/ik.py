"""Inverse-kinematics helpers for MuJoCo models."""

import mujoco
import numpy as np


def solve_position_ik(
    model: mujoco.MjModel,
    workspace: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int = 30,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> np.ndarray:
    """Levenberg-Marquardt IK for a site position."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(max_iters):
        workspace.qpos[: model.nq] = q
        workspace.qvel[:] = 0.0
        mujoco.mj_forward(model, workspace)
        err = target_pos - workspace.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, workspace, jacp, None, site_id)
        JJ = jacp @ jacp.T + damping * np.eye(3)
        dq = jacp.T @ np.linalg.solve(JJ, err)
        q += dq
    return q
