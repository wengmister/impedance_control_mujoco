"""
Visualization script for xArm7 (no hand) model using MuJoCo interactive GUI.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

from util.torque_viz import TorqueIndicator
from util.traj_viz import TrajectoryTrail


MODEL_DIR = Path("mujoco_menagerie/ufactory_xarm7")
MODEL_XML = MODEL_DIR / "scene_passive.xml"
EE_SITE = "attachment_site"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="xArm7 visualization")
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Number of samples for EE trail (0 disables)",
    )
    return parser.parse_args()


def main():
    """Main visualization function using interactive MuJoCo GUI."""
    args = parse_args()

    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Initialize physics
    mujoco.mj_forward(model, data)

    print(f"Model loaded successfully!")
    print(f"  - Bodies: {model.nbody}")
    print(f"  - Joints: {model.njnt}")
    print(f"  - Actuators: {model.nu}")
    print(f"  - Keyframes: {model.nkey}")
    print("\nStarting interactive visualization...")
    print("Use the mouse to interact with the model in the viewer.")

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

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            if trail is not None and trail_site is not None:
                trail.update(data.site_xpos[trail_site])

            # Step simulation
            mujoco.mj_step(model, data)

            ngeom = torque_indicator.draw(viewer, model, data, base_index=0)
            if trail is not None:
                ngeom = trail.draw(viewer, base_index=ngeom)
            else:
                viewer.user_scn.ngeom = ngeom

            # Update viewer
            viewer.sync()


if __name__ == "__main__":
    main()
