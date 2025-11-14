# Notes

## `CTRLRANGE` vs `FORCERANGE`

- Torque motors were saturating because `ctrlrange` stayed at MuJoCo’s default ±1 while the desired torques were ±20–50 Nm.
- Matching `ctrlrange` to `forcerange` in `xarm7_passive.xml` fixes gravity compensation; keep them aligned unless you intentionally want to limit commanded torque.


## Mujoco Affine Actuators

Affine actuators (`biastype="affine"`) compute force/torque as:

$$
\tau = k * u + b0 + b1 * qdot + b2 * q
$$

Where:

- `gainprm = [k]`  
  The linear gain on the control input `u` (only 1 parameter).

- `biasprm = [b0, b1, b2]`  
  These are *state multipliers*, not simple biases.  
  - `b0`: constant offset  
  - `b1`: velocity coefficient (adds viscous damping)  
  - `b2`: position coefficient (adds a spring term centered at q = 0)

Important points:

- `biasprm[1]` and `biasprm[2]` implicitly create a PD controller centered at zero:
  - `b1 < 0` adds stabilizing damping
  - `b2 < 0` pulls the joint toward q = 0
- Because of this, the actuator may behave as if it has built-in impedance even if you intend pure torque control.
- For true torque control, set:
  
      <general biastype="affine" gainprm="1" biasprm="0 0 0"/>

- The name “bias” is misleading: in MuJoCo it refers to *any* state-dependent contribution, not just a constant offset.
