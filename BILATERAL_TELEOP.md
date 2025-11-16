# Bilateral Teleoperation

teleoperating a master robot arm (B) controlling a slave arm (A) with **haptic feedback**.

---

## **1. Joint-Space Bilateral Impedance (Best when the arms are similar)**

**Idea:**
Connect corresponding joints with a virtual *spring–damper*.

**Control law:**

* Coupling torque on A:
  $$ \tau_A^{\text{coup}} = K (q_B - q_A) + D (\dot q_B - \dot q_A) $$
* Coupling torque on B:
  $$ \tau_B^{\text{coup}} = -\left[ K (q_B - q_A) + D (\dot q_B - \dot q_A) \right] $$

**Total torques:**

* $ \tau_A = \tau_A^{\text{grav}} + \tau_A^{\text{coup}} $
* $ \tau_B = \tau_B^{\text{grav}} + \tau_B^{\text{coup}} $

**Behavior:**

* Move B → spring pulls A.
* A hits environment → $q_A$ lags → spring stretches → operator feels resistive torque in B.

**Pros:**

* Simple, responsive, minimal math.

**Cons:**

* Only works well if kinematics & joint ranges are similar.
* Joint mapping becomes ambiguous if DoF differ.

---

## **2. Task-Space Bilateral Control (Different kinematics OK)**

**Step 1 — Map master motion to slave motion**

* Let $x_B$ = EEF pose of master.
* Define $x_A^{\text{des}} = f(x_B)$ (scale, offset, constraints, etc.).

Slave A uses task-space impedance:

$$
F_v = K_p (x_A^{\text{des}} − x_A) + K_d (\dot x_A^{\text{des}} − \dot x_A)
$$

Joint torques on A:

$$
τ_A = J_A^T F_v + τ_A^{grav}
$$

**Step 2 — Reflect environment forces back to B**

Given environment wrench $F_{\text{env},A}$ at the slave:

$$
τ_B = J_B^T ( S · F_{\text{env},A} ) + τ_B^{grav}
$$

Where:

* $S$ = scaling/filtering matrix
* $J_A$, $J_B$ = Jacobians of A and B

**Behavior:**

* Operator moves B → A tracks via impedance in Cartesian space.
* A contacts objects → $F_{\text{env}}$ pushes back → reflected through $J_B^T$ as haptic feedback.

**Pros:**

* Works when DoF mismatch (6-DoF vs 7-DoF, etc.).
* Task-space semantics preserved.

**Cons:**

* Requires Jacobians, IK, decent torque authority.
* Needs force sensing or observer on A.

---

## **3. Virtual Proxy**

**Idea:**
Introduce an intermediate “proxy” pose $x_P$ between master and slave.
This creates a *passive stable coupling* instead of hard binding.

Master B → Proxy coupling:

$$
F_B = K_B (x_P − x_B) + D_B (\dot{x_P} − \dot{x_B})
$$

Slave A → Proxy tracking:

$$
F_A^{cmd} = K_A (x_P − x_A) + D_A (\dot{x_P} − \dot{x_A})
τ_A = J_A^T F_A^{cmd} + τ_A^{grav}
$$

Proxy interacts with environment through $F_{\text{env},A}$.

Force reflection back to B:

$$
τ_B = J_B^T ( S · F_{\text{env},A} ) + τ_B^{grav}
$$

**Behavior:**

* Proxy acts like a “virtual spring box” between B and A.
* Stabilizes contact transitions and delays.
* Lets you tune “feel” independently on master/slave.

**Pros:**

* Most stable architecture (used in real haptic systems).
* Smooth and tunable.
* Handles discontinuous contacts well.

**Cons:**

* Requires more tuning and an internal proxy update rule.

---

# **When To Use Which**

| Scenario                                       | Recommended Approach                                 |
| ---------------------------------------------- | ---------------------------------------------------- |
| Two similar arms, similar joint ranges         | Joint-space bilateral impedance                      |
| Arms with different DoF / kinematics           | Task-space bilateral control                         |
| **Need stability under contact / latency / noise** | **Virtual proxy (virtual coupling)**                     |
| No torque control on master                    | Admittance-like variation of task-space mapping      |
| Have F/T sensor only on slave                  | All methods supported (force reflection via `$J^T$`) |

---

# **Key Mapping Concepts (important when kinematics differ)**

* Represent motion and forces in **task space**: position, orientation, twist, and wrench.
* Use a mapping function:
  $ x_A^{des} = f(x_B) $
* Convert slave task-space forces to master torques using:
 $ τ_B = J_B^T F_B $
* Convert master motion to slave task-space commands via inverse kinematics.

