# Admittance Control
## **1. Core Definitions**

* **Impedance control**

  * External force affects robot **torque response**.
  * Controller enforces a *desired dynamic relationship*:
    $$
    M_v \ddot{x} + B_v(\dot{x}-\dot{x}*d) + K_v(x-x_d) = F*{\text{ext}}
    $$
  * Robot physically behaves like a mass–spring–damper.
  * Requires decent inverse dynamics or torque control.

* **Admittance control**

  * External force affects **motion commands** (position/velocity).
  * Robot does **not** behave like the dynamic system — the *virtual model* does.

---

## **2. Admittance ODE Integration**

* Virtual model:
  $$
  M_v \ddot{x}*d + B_v \dot{x}*d + K_v (x_d - x*{\text{ref}}) = F*{\text{ext}}
  $$

* Solve for acceleration:
  $$
  \ddot{x}*d = M_v^{-1}\big(F*{\text{ext}} - B_v \dot{x}*d - K_v(x_d - x*{\text{ref}})\big)
  $$

* Discrete-time integration:
  $$
  \begin{aligned}
  \dot{x}_d[k+1] &= \dot{x}_d[k] + \ddot{x}_d[k]\Delta t \
  x_d[k+1] &= x_d[k] + \dot{x}_d[k+1]\Delta t
  \end{aligned}
  $$

* Final output:

  * Desired position/velocity sent to robot's position controller.
  * Works even on robots without torque control.

---

## **3. Simplified Admittance (Force → Velocity or Position)**

* Common practical variant:
  $$
  \dot{x}*d = \alpha F*{\text{ext}}
  $$

* Or with torques:
  $$
  \dot{\theta}*d = k*{\tau},\tau_{\text{ext}}
  $$

* Properties:

  * No ODE solving.
  * Behaves like **first-order admittance** (zero virtual mass).
  * Used in “force-to-velocity”, “hand-guiding”, “compliance” modes.
  * Still admittance because *force inputs generate motion outputs*.

---

## **4. Why Use Admittance Even on Torque-Controlled Robots**

* **Robust to friction and unmodeled dynamics**

  * Impedance depends on dynamic model accuracy.
  * Admittance bypasses friction, inertia, Coriolis modeling entirely.

* **Easier tuning & guaranteed stability**

  * High damping → no oscillations.
  * Position loop handles low-level stability.

* **Preferred for hand-guiding / teaching by demonstration**

  * Smooth, predictable motion.
  * No physical inertia or “springiness.”

* **More predictable than impedance for low stiffness**

  * Impedance can destabilize with:

    * low stiffness
    * low damping
    * inaccurate inverse dynamics

* **Used by real robots even when torque sensing exists**

  * Franka “compliance” and “hand guiding”
  * KUKA LBR iiwa guiding
  * UR “force mode”
  * Many of these are *admittance under the hood*.

---

## **5. How Admittance Feels Compared to Impedance**

* Behaviorally:

  * Admittance *feels like* a very overdamped impedance controller.
  * Both produce:

    * smooth motion
    * high damping
    * no oscillation
    * viscous response

* But internally:

  * **Impedance:** enforces physical dynamics via torque.
  * **Admittance:** generates motion from a virtual model.

* So the feeling is similar, but the mechanism is fundamentally different.

---

## **6. Impedance vs Admittance — Quick Comparison Table**

| Aspect                      | Impedance Control   | Admittance Control              |
| --------------------------- | ------------------- | ------------------------------- |
| Input                       | Force → *torque*    | Force → *motion*                |
| Robot must be               | Torque-controllable | Position-/velocity-controllable |
| Dynamics                    | Robot obeys MSD     | Virtual MSD produces motion     |
| Sensitivity to model errors | High                | Low                             |
| Handles friction            | Poorly              | Very well                       |
| Stability at low stiffness  | Tricky              | Easy                            |
| Typical use                 | Contact tasks       | Hand guiding, teaching          |
| Behavior feel               | Springy, dynamic    | Smooth, overdamped              |

---

## **7. Joint-Level Admittance**

* Even with joint torque sensors:
  $$
  \dot{q}*d = k*\tau , \tau_{\text{ext}}
  $$
  $$
  q_d[k+1] = q_d[k] + \dot{q}_d[k]\Delta t
  $$

* Reasons to use joint admittance:

  * Compensates friction automatically.
  * Produces very smooth guiding.
  * Robust to model errors.
  * Often safer and simpler than full joint impedance.

---

Admittance control converts external forces or torques into motion commands by simulating (or simplifying) a virtual compliant model.

Impedance control converts external forces into torques so the robot physically behaves like that compliant model.
In practice, admittance often feels like an overdamped impedance controller, but internally they are fundamentally different.