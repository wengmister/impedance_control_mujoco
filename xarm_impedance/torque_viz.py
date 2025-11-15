"""Render per-joint torque indicators in the MuJoCo viewer."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class TorqueIndicator:
    """Visualize actuator effort with cylindrical bars next to each joint."""

    scale: float = 0.1  # bar length in meters at 100% effort
    radius: float = 0.02  # bar radius in meters

    def _joint_position(
        self, model: mujoco.MjModel, data: mujoco.MjData, joint_id: int
    ) -> np.ndarray:
        """Compute world-space position of a joint."""
        body_id = model.jnt_bodyid[joint_id]
        body_pos = data.xpos[body_id]
        body_mat = data.xmat[body_id].reshape(3, 3)
        local = model.jnt_pos[joint_id]
        return body_pos + body_mat @ local

    def _rotation_from_direction(self, direction: np.ndarray) -> np.ndarray:
        """Build rotation matrix with given direction as Z-axis."""
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(up, direction)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, direction)
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_axis = np.cross(direction, x_axis)
        return np.column_stack((x_axis, y_axis, direction)).reshape(9)

    def draw(
        self,
        viewer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_index: int = 0,
    ) -> int:
        """Draw torque bars for all actuated joints."""
        user_scn = viewer.user_scn
        user_scn.ngeom = base_index
        max_geoms = len(user_scn.geoms)

        for j in range(model.nv):
            if user_scn.ngeom >= max_geoms:
                break

            joint_pos = self._joint_position(model, data, j)

            # Normalize effort by actuator force range
            effort = data.qfrc_actuator[j] if j < model.nu else 0.0
            max_force = model.actuator_forcerange[j, 1] if j < model.nu else 1.0
            fraction = np.clip(abs(effort) / max_force, 0.0, 1.0)

            # Bar length scales with effort fraction
            length = max(1e-4, self.scale * fraction)
            half_length = length / 2.0

            # Color gradient
            color = np.array(
                [fraction, 1.0 - fraction, 0.2, 0.7], dtype=np.float32
            )

            # Cylinder size: [radius, half_length, 0]
            size = np.array([self.radius, half_length, 0.0], dtype=np.float64)

            # Position along local Z-axis with small offset
            offset_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            body_rot = data.xmat[model.jnt_bodyid[j]].reshape(3, 3)
            offset_dir = body_rot @ offset_dir
            pos = joint_pos + offset_dir * (half_length + 0.05)

            geom_rot = self._rotation_from_direction(offset_dir)

            mujoco.mjv_initGeom(
                user_scn.geoms[user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                size,
                pos,
                geom_rot,
                color,
            )
            user_scn.ngeom += 1

        return user_scn.ngeom
