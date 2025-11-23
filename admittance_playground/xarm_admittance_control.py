"""
Task-space admittance controller V2 for xArm7.

Implements a velocity-based admittance control law:
    v_cmd = K * (F_ext - F_deadband)
    x_cmd = x_prev + v_cmd * dt

This allows "push-and-stay" behavior.
"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from util.ik import solve_position_ik
from util.torque_viz import TorqueIndicator
from util.traj_viz import TrajectoryTrail


MODEL_XML = Path("admittance_playground/scene/scene_xarm_admittance.xml")
EE_SITE = "attachment_site"

# Keyframe pose from the MJCF
HOME_QPOS = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])

# Admittance V2 Parameters
ADMITTANCE_GAIN = 0.01    # (m/s) / N - Velocity per unit force
DEADBAND = 2.0            # N - Ignore forces below this magnitude
MAX_VELOCITY = 0.5        # m/s - Safety limit
DT = 0.002                # s - Control loop timestep (approx)
FORCE_SIGN = -1.0          # Flip direction to align force and movement

class AdmittanceController:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_id: int,
        force_sensor_id: int,
        torque_sensor_id: int,
        q_home: np.ndarray,
    ):
        self.model = model
        self.site_id = site_id
        self.force_sensor_id = force_sensor_id
        self.torque_sensor_id = torque_sensor_id

        self.force_sensor_adr = model.sensor_adr[force_sensor_id]
        self.torque_sensor_adr = model.sensor_adr[torque_sensor_id]

        # IK Workspace
        self.ik_workspace = mujoco.MjData(model)
        
        # Initialize state
        self.q_des = q_home.copy()
        
        # Get initial EE position from q_home
        self.ik_workspace.qpos[:model.nq] = q_home
        mujoco.mj_forward(model, self.ik_workspace)
        self.x_target = self.ik_workspace.site_xpos[site_id].copy()

    def update(self, data: mujoco.MjData) -> np.ndarray:
        # 1. Read Sensors
        force_sensor = data.sensordata[self.force_sensor_adr : self.force_sensor_adr + 3]
        # Transform to world frame
        R_site = data.site_xmat[self.site_id].reshape(3, 3)
        F_ext = FORCE_SIGN * (R_site @ force_sensor)

        # 2. Apply Deadband
        F_mag = np.linalg.norm(F_ext)
        if F_mag < DEADBAND:
            F_cmd = np.zeros(3)
        else:
            # Scale force, keeping direction
            F_cmd = F_ext * (1.0 - DEADBAND / F_mag)

        # 3. Compute Velocity Command
        v_cmd = F_cmd * ADMITTANCE_GAIN
        
        # Clip velocity for safety
        v_mag = np.linalg.norm(v_cmd)
        if v_mag > MAX_VELOCITY:
            v_cmd = v_cmd * (MAX_VELOCITY / v_mag)

        # 4. Integrate Position
        self.x_target += v_cmd * DT

        # 5. Solve IK
        # Use current q_des as warm start to ensure continuity
        self.q_des = solve_position_ik(
            self.model,
            self.ik_workspace,
            self.site_id,
            self.x_target,
            self.q_des, # Warm start
            max_iters=10,
            tol=1e-4,
        )

        return self.q_des


def main():
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_XML}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Start from home
    data.qpos[:model.nq] = HOME_QPOS
    mujoco.mj_forward(model, data)
    
    # Get IDs
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
    torque_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque")
    
    if ee_site_id < 0: raise ValueError(f"Site {EE_SITE} not found")
    
    # Controller
    controller = AdmittanceController(
        model, ee_site_id, force_sensor_id, torque_sensor_id, HOME_QPOS
    )
    
    # Viz
    torque_indicator = TorqueIndicator(scale=0.1)
    
    print("Launching Admittance Controller V2...")
    print("  - Velocity-based (Push and Stay)")
    print("  - High Joint Stiffness")
    print("\nUse Ctrl+Left-click to drag the blue sphere!")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Update Controller
            q_des = controller.update(data)
            
            # Send Command
            data.ctrl[:] = q_des
            
            # Step Sim
            mujoco.mj_step(model, data)
            
            # Viz
            viewer.user_scn.ngeom = torque_indicator.draw(viewer, model, data, base_index=0)
            viewer.sync()

            # Timing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
