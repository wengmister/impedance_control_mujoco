# Impedance Control - Background

This document summarizes the two formulations of impedance control:

1. **Force-based impedance** (simple, no inertia shaping)
2. **Acceleration-based impedance** (full inertia shaping)

Each in:
- **Joint space**
- **Task space**

And implementation notes for **MuJoCo**.

---

# 0. Robot Dynamics

Joint-space dynamics:

$$
M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau + \tau_{\mathrm{ext}}
$$

Task-space kinematics:

$$
\dot{x} = J(q)\dot{q}
$$

$$
\ddot{x} = J(q)\ddot{q} + \dot{J}(q,\dot{q})\dot{q}
$$

Task-space inertia:

$$
\Lambda(q) = \left(J M^{-1} J^\top\right)^{-1}
$$

---

# 1. Formulation A — Force-Based Impedance
This version **does not shape inertia**.  
You directly output a force/wrench from a virtual spring-damper.

## 1.1 Joint-Space Force-Based Impedance

Desired behavior:

$$
D_q(\dot{q} - \dot{q}_d) + K_q(q - q_d) \approx \tau_{\mathrm{ext}}
$$

Controller:

$$
\tau = -K_q (q - q_d) - D_q (\dot{q} - \dot{q}_d)
$$

Model-aided version:

$$
\tau = -K_q (q - q_d) - D_q (\dot{q} - \dot{q}_d) + g(q)
$$

---

## 1.2 Task-Space Force-Based Impedance

Desired behavior:

$$
D_x(\dot{x} - \dot{x}_d) + K_x(x - x_d) \approx F_{\mathrm{ext}}
$$

Controller:

$$
F_{\mathrm{cmd}} = -K_x (x - x_d) - D_x (\dot{x} - \dot{x}_d)
$$

Joint torques:

$$
\tau = J^\top F_{\mathrm{cmd}} + g(q)
$$

No inertia shaping occurs here.

---

# 2. Formulation B — Acceleration-Based Impedance
This version **shapes the apparent inertia**.  
You output a **desired acceleration**, then use **inverse dynamics** to enforce it.

---

## 2.1 Joint-Space Acceleration-Based Impedance

Desired impedance:

$$
M_d(\ddot{q} - \ddot{q}_d) +
D_d(\dot{q} - \dot{q}_d) +
K_d(q - q_d)
= \tau_{\mathrm{ext}}
$$

Solve for desired acceleration:

$$
\ddot{q}_{cmd}
= \ddot{q}_d
- M_d^{-1}
\left[
D_d(\dot{q} - \dot{q}_d)
+ K_d(q - q_d)
- \tau_{\mathrm{ext}}
\right]
$$

Then use inverse dynamics:

$$
\tau = M(q)\ddot{q}_{cmd} + C(q,\dot{q}) + g(q)
$$

This requires the **full inverse dynamics**, because inertia shaping works by explicitly injecting the correct $M$, $C$, and $g$ terms.

---

## 2.2 Task-Space Acceleration-Based Impedance (Operational Space)

Desired impedance:

$$
M_d(\ddot{x} - \ddot{x}_d)
+ D_d(\dot{x} - \dot{x}_d)
+ K_d(x - x_d)
= F_{\mathrm{ext}}
$$

Desired acceleration:

$$
\ddot{x}_{cmd}
= \ddot{x}_d
- M_d^{-1}
\left[
D_d(\dot{x} - \dot{x}_d)
+ K_d(x - x_d)
- F_{\mathrm{ext}}
\right]
$$

Operational-space inverse dynamics:

$$
F_{\mathrm{cmd}}
= \Lambda \ddot{x}_{cmd} + \mu(q,\dot{q}) + p(q)
$$

Joint torques:

$$
\tau = J^\top F_{\mathrm{cmd}}
$$

This shapes the apparent Cartesian inertia to match $M_d$.

---

# 3. Implementation in MuJoCo

Below are the practical steps for each controller type.

---

## 3.1 MuJoCo: Force-Based Joint-Space Impedance

1. Read joint state:
   ```python
   q = data.qpos
   qd = data.qvel
   ```

2. Compute torque:

    ```python
    tau = -Kq @ (q - q_des) - Dq @ (qd - qd_des)
    ```


3. Add gravity compensation (optional):

    ```python
    g = compute_gravity_rne(...)
    tau += g
    ```


4. Apply:

    ```python
    data.ctrl[:] = tau
    ```

## 3.2 MuJoCo: Force-Based Task-Space Impedance

1. Compute end-effector pose:

    ```
    x = data.xpos[ee_body]
    ```


2. Compute Jacobian:


    ```
    Jp, Jr = mj_jacBody(...)
    ```

3. Task-space errors:

    ```
    ex = x - x_des
    edot = J @ qd
    ```

4. Command wrench:

    ```
    F_cmd = -Kx @ ex - Dx @ edot
    ```

5. Joint torques:

    ```
    tau = J.T @ F_cmd + g
    ```

6. Apply to data.ctrl.

## 3.3 MuJoCo: Acceleration-Based Joint-Space Impedance

Compute desired $\ddot{q}_{cmd}$:

```python
qdd_cmd = qdd_des - inv(Md) @ (Dd @ ed + Kd @ e - tau_ext_est)
```

Use RNE for inverse dynamics:

data.qacc[:] = qdd_cmd
mujoco.mj_rne(model, data, ...)
tau = data.qfrc_inverse


Apply.

## 3.4 MuJoCo: Acceleration-Based Task-Space Impedance

Compute $x$, $\dot{x}$, and $J$.

Compute $\ddot{x}_{cmd}$.

Get $\Lambda$, $\mu$, $p$ from mass matrix APIs or RNE.

Compute:

```python
F_cmd = Lambda @ xdd_cmd + mu + p
tau = J.T @ F_cmd
```

Apply.