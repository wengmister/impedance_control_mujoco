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

## MuJoCo `mj_rne` ID

### What `mj_rne` computes
`mj_rne` performs Recursive Newton–Euler inverse dynamics.

- With `flg_acc = 0`:
  
    $$
      tau = C(q, qdot) + g(q)
    $$

  (bias forces only; ignores qacc)

- With `flg_acc = 1`:
  
    $$
      tau = M(q) * qacc + C(q, qdot) + g(q)
    $$

  (full inverse dynamics; uses qacc)

### Meaning of `flg_acc`
- `flg_acc = 0` → compute only Coriolis + gravity (assume desired acceleration = 0)
- `flg_acc = 1` → include inertia term based on `data.qacc` (for desired accelerations)

### When to use each mode
- Use `flg_acc = 0` when:
  - doing gravity compensation
  - holding a static posture
  - passive/floating behavior
  - you have **no desired acceleration**

- Use `flg_acc = 1` when:
  - tracking a trajectory
  - using PD control converted to desired qacc
  - computed torque / inverse dynamics control
  - operational-space or impedance control
  - you **have a desired acceleration**

### Role of `data.qacc`
- For `flg_acc = 0`, `qacc` is ignored; it is common to set `data.qacc[:] = 0`.
- For `flg_acc = 1`, `qacc` **must** contain your desired acceleration; do **not** zero it.

### Relationship to `qfrc_bias`
- `mj_rne(flg_acc = 0)` returns:

    $$
      C + g
    $$

  (rigid-body bias forces only)

- `data.qfrc_bias` contains:

    $$
      C + g + passive forces
    $$

  (includes damping, actuator bias, springs, friction, soft constraints)

Thus `mj_rne` gives clean rigid-body dynamics; `qfrc_bias` includes additional model-specific passive effects.

### Summary
- `flg_acc = 0` → use when desired acceleration is zero. Good for gravity comp.
- `flg_acc = 1` → use when you compute a desired acceleration. Needed for proper inverse dynamics.
- Zeroing `qacc` only makes sense when you truly want qacc_des = 0.

