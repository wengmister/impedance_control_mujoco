"""
Gravity compensation controller for the passive xArm7 MuJoCo model.
"""

import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from util.torque_viz import TorqueIndicator
from util.traj_viz import TrajectoryTrail


MODEL_DIR = Path("mujoco_menagerie/franka_fr3")
MODEL_XML = MODEL_DIR / "fr3_passive.xml"
HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
EE_SITE = "attachment_site"


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return C(q, qd) + g(q) via Recursive Newton-Euler."""
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="franka gravity compensation demo")
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Number of samples for EE trajectory trail (0 disables)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not foun   d: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print("Launching Franka gravity compensation demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  actuator ctrlrange:\n{model.actuator_ctrlrange}")
    print(f"  actuator forcerange:\n{model.actuator_forcerange}")

    trail = (
        TrajectoryTrail(maxlen=args.trail_length, stride=25)
        if args.trail_length > 0
        else None
    )
    trail_site = None
    if trail is not None:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if site_id < 0:
            raise ValueError(f"Site '{EE_SITE}' not found in model.")
        trail_site = site_id
    torque_indicator = TorqueIndicator(scale=0.12)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if trail is not None and trail_site is not None:
                trail.update(data.site_xpos[trail_site])
            bias = compute_bias_forces(model, data)
            data.ctrl[:] = bias
            if step % 50 == 0:
                print(
                    f"[step {step}] bias min/max: {bias.min(): .2f} / {bias.max(): .2f}, "
                    f"norm: {np.linalg.norm(bias): .2f}"
                )
            mujoco.mj_step(model, data)
            ngeom = torque_indicator.draw(viewer, model, data, base_index=0)
            if trail is not None:
                ngeom = trail.draw(viewer, base_index=ngeom)
            else:
                viewer.user_scn.ngeom = ngeom
            viewer.sync()
            step += 1


if __name__ == "__main__":
    main()
