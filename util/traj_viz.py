"""Utilities for visualizing end-effector trajectories in the MuJoCo viewer."""

from collections import deque
from typing import Optional

import mujoco
import numpy as np


class TrajectoryTrail:
    """Ring-buffer trail rendered as semi-transparent spheres."""

    def __init__(
        self,
        maxlen: int,
        stride: int = 1,
        radius: float = 0.004,
        color: Optional[np.ndarray] = None,
    ) -> None:
        self.enabled = maxlen > 0
        self._points = deque(maxlen=maxlen) if self.enabled else None
        self._stride = max(1, stride)
        self._radius = radius
        self._steps_since = 0
        base_rgba = np.array([0.9, 0.2, 0.2, 0.6], dtype=np.float32)
        if color is not None:
            base_rgba = color.astype(np.float32)
        self._base_rgba = base_rgba
        self._size = np.array([radius, 0.0, 0.0], dtype=np.float64)
        self._identity = np.eye(3, dtype=np.float64).reshape(9)

    def update(self, position: np.ndarray) -> None:
        if not self.enabled or self._points is None:
            return
        self._steps_since += 1
        if self._steps_since < self._stride:
            return
        self._steps_since = 0
        self._points.append(np.asarray(position, dtype=np.float64).copy())

    def draw(self, viewer, base_index: int = 0) -> int:
        if not self.enabled or self._points is None:
            viewer.user_scn.ngeom = base_index
            return base_index
        user_scn = viewer.user_scn
        user_scn.ngeom = base_index
        max_geoms = len(user_scn.geoms)
        total = len(self._points)
        if total == 0:
            return base_index
        for idx, pos in enumerate(self._points):
            if user_scn.ngeom >= max_geoms:
                break
            geom = user_scn.geoms[user_scn.ngeom]
            alpha = (idx + 1) / total
            rgba = self._base_rgba.copy()
            rgba[3] *= alpha
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                self._size,
                pos,
                self._identity,
                rgba,
            )
            user_scn.ngeom += 1
        return user_scn.ngeom
