"""
Visualization script for xArm7 (no hand) model using MuJoCo interactive GUI.
"""

import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer


def main():
    """Main visualization function using interactive MuJoCo GUI."""
    # Path to the xarm7_nohand model
    model_dir = Path("mujoco_menagerie/ufactory_xarm7")
    model_xml = model_dir / "scene_nohand.xml"
    
    
    # Load model
    print(f"Loading model from {model_xml}...")
    if not model_xml.exists():
        raise FileNotFoundError(f"Model file not found: {model_xml}")
    
    model = mujoco.MjModel.from_xml_path(str(model_xml))
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
    
    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()


if __name__ == "__main__":
    main()
