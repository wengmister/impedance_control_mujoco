"""
Gravity compensation controller for the passive xArm7 MuJoCo model.
"""

from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


MODEL_DIR = Path("mujoco_menagerie/ufactory_xarm7")
MODEL_XML = MODEL_DIR / "scene_passive.xml"
HOME_QPOS = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])


def compute_bias_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return C(q, qd) + g(q) via Recursive Newton-Euler."""
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc
    return bias


def main():
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not foun   d: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    data.qpos[: model.nq] = HOME_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print("Launching xArm7 gravity compensation demo...")
    print(f"  model: {MODEL_XML}")
    print(f"  actuator ctrlrange:\n{model.actuator_ctrlrange}")
    print(f"  actuator forcerange:\n{model.actuator_forcerange}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            bias = compute_bias_forces(model, data)
            data.ctrl[:] = bias
            if step % 50 == 0:
                print(
                    f"[step {step}] bias min/max: {bias.min(): .2f} / {bias.max(): .2f}, "
                    f"norm: {np.linalg.norm(bias): .2f}"
                )
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1


if __name__ == "__main__":
    main()
