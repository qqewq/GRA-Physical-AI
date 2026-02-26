```markdown
# Projectors, Goals, and Foam: The Mathematics of Consistency

[< back to Documentation](../README.md) | [previous: layers.md](layers.md) | [next: algorithm.md](../architecture/algorithm.md)

In the GRA framework, every **goal** is represented by a **projector** – a mathematical object that picks out the subspace of states satisfying that goal.  
Inconsistencies between subsystems are quantified by **foam**, which measures how much the projectors of different subsystems fail to align.

This document provides a detailed exposition of projectors, their properties, and how they are used to define goals and foam.  
We also discuss practical implementation aspects for physical AI systems.

---

## 1. Goals as Subspaces

A **goal** \(G\) for a system is a condition that the system’s state should satisfy.  
In the Hilbert space \(\mathcal{H}\) of all possible states, the set of states that satisfy the goal forms a **subspace** \(\mathcal{H}_G \subseteq \mathcal{H}\).

**Examples**:
- For a camera: “calibrated” → subspace of images with known calibration parameters.
- For a motor: “within safe torque limits” → subspace of torque values ≤ limit.
- For a planner: “collision‑free trajectory” → subspace of trajectories that avoid obstacles.
- For an ethical rule: “do not harm” → subspace of actions that cause no harm.

---

## 2. Projectors: Definition and Properties

### 2.1. Orthogonal Projector

An **orthogonal projector** (or simply **projector**) onto a subspace \(\mathcal{H}_G\) is a linear operator \(\mathcal{P}_G : \mathcal{H} \to \mathcal{H}\) such that:

- \(\mathcal{P}_G^2 = \mathcal{P}_G\) (idempotent).
- \(\mathcal{P}_G^\dagger = \mathcal{P}_G\) (self‑adjoint).
- \(\operatorname{Range}(\mathcal{P}_G) = \mathcal{H}_G\).
- \(\operatorname{Ker}(\mathcal{P}_G) = \mathcal{H}_G^\perp\).

For any state \(|\psi\rangle \in \mathcal{H}\), the projected state \(\mathcal{P}_G |\psi\rangle\) is the **closest** state in \(\mathcal{H}_G\) to \(|\psi\rangle\) (in the Hilbert space norm).  
The quantity \(\| (I - \mathcal{P}_G) |\psi\rangle \|^2\) measures how far \(|\psi\rangle\) is from satisfying the goal.

### 2.2. Spectral Representation

If we choose an orthonormal basis \(\{ |e_i\rangle \}\) of \(\mathcal{H}_G\) and extend to a full basis of \(\mathcal{H}\), then:

\[
\mathcal{P}_G = \sum_i |e_i\rangle\langle e_i|.
\]

### 2.3. Important Properties

- **Positivity**: \(\langle \psi | \mathcal{P}_G | \psi \rangle \ge 0\) for all \(|\psi\rangle\).
- **Eigenvalues**: 1 (on the subspace) and 0 (on the orthogonal complement).
- **Commutativity**: Two projectors \(\mathcal{P}_A\) and \(\mathcal{P}_B\) commute iff the subspaces are **compatible** in the sense that they can be simultaneously diagonalized. This is equivalent to \(\mathcal{P}_A \mathcal{P}_B\) being itself a projector (onto the intersection of the subspaces).

---

## 3. Goals in GRA

In a GRA multiverse, each subsystem identified by a multi‑index \(\mathbf{a}\) (with \(\dim(\mathbf{a}) = l\)) has its own goal \(G_l^{(\mathbf{a})}\), represented by a projector \(\mathcal{P}_{G_l^{(\mathbf{a})}}\) acting on \(\mathcal{H}^{(\mathbf{a})}\).

### 3.1. Level‑wise Projectors

For a given level \(l\), the collection of projectors for all subsystems at that level defines a **global projector** on the level‑\(l\) subspace \(\mathcal{H}^{(l)} = \bigotimes_{\dim(\mathbf{a})=l} \mathcal{H}^{(\mathbf{a})}\):

\[
\mathcal{P}_{G_l} = \bigotimes_{\dim(\mathbf{a})=l} \mathcal{P}_{G_l^{(\mathbf{a})}} .
\]

This is consistent with the **hierarchy consistency condition** ([theorems.md](theorems.md)): the goal of a higher‑level system is the tensor product of the goals of its subsystems.

### 3.2. Examples from Physical AI

| Level | Subsystem | Goal | Projector action |
|-------|-----------|------|------------------|
| G₀ | Camera | Calibrated | Projects onto images with known calibration parameters. |
| G₀ | Motor | Torque within limits | Projects onto torque vectors with each component ≤ limit. |
| G₁ | Vision system | Time‑synchronized sensors | Projects onto states where all camera timestamps match within a tolerance. |
| G₂ | Planner | Collision‑free trajectory | Projects onto trajectories that avoid all obstacles (as predicted by world model). |
| G₃ | Dialogue | Truthful responses | Projects onto utterances consistent with the robot’s internal state. |
| G₄ | Ethics | Do no harm | Projects onto actions that do not cause physical/psychological harm. |

---

## 4. Commutativity and Consistency

### 4.1. Why Commutativity Matters

The **Multiverse Zeroing Theorem** requires that all projectors commute:

\[
[\mathcal{P}_{G_l^{(\mathbf{a})}}, \mathcal{P}_{G_m^{(\mathbf{b})}}] = 0 \quad \forall \mathbf{a},\mathbf{b},\; l,m.
\]

If projectors do not commute, there is **no basis** in which all goals are simultaneously diagonalized. This means that no single state can perfectly satisfy all goals – some trade‑off is inevitable.  
In physical terms, non‑commuting goals represent **fundamental conflicts** (e.g., “minimize energy” and “maximize speed” may be incompatible).  
In such cases, full zeroing is impossible; we must either relax goals or redesign them to commute.

### 4.2. Hierarchy Consistency Condition

For the hierarchy to be well‑formed, we also require:

\[
\mathcal{P}_{G_l^{(\mathbf{a})}} = \bigotimes_{\mathbf{b} \prec \mathbf{a}} \mathcal{P}_{G_{l-1}^{(\mathbf{b})}} \quad \forall l \ge 1.
\]

This ensures that satisfying lower‑level goals is **sufficient** to satisfy the higher‑level goal (up to additional cross‑constraints captured by foam).  
If this condition fails, the higher‑level goal imposes extra requirements not reducible to lower levels, breaking the recursive structure.

---

## 5. Foam – Quantifying Inconsistency

### 5.1. Definition

For a fixed level \(l\), let \(\Psi^{(l)} = \{ \Psi^{(\mathbf{a})} \}_{\dim(\mathbf{a})=l}\) be the collection of states of all subsystems at that level.  
The **foam** at level \(l\) is defined as:

\[
\Phi^{(l)}(\Psi^{(l)}, G_l) = \sum_{\substack{\mathbf{a}\neq\mathbf{b} \\ \dim(\mathbf{a})=\dim(\mathbf{b})=l}} \bigl| \langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle \bigr|^2 .
\]

### 5.2. Interpretation

- The inner product \(\langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle\) measures the **overlap** between the state of subsystem \(\mathbf{a}\) and the goal‑satisfying part of subsystem \(\mathbf{b}\).
- If the two subsystems are **consistent**, their states should be **independent** with respect to the common goal \(G_l\). In the eigenbasis of \(\mathcal{P}_{G_l}\), this means that off‑diagonal elements vanish.
- Hence, foam is the **sum of squared off‑diagonal elements** – a measure of how much the subsystems “interfere” with each other’s goal satisfaction.

### 5.3. Zero Foam

Zero foam at level \(l\) implies that for all distinct \(\mathbf{a},\mathbf{b}\),

\[
\langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle = 0 .
\]

In the eigenbasis of \(\mathcal{P}_{G_l}\), this means that each \(\Psi^{(\mathbf{a})}\) is an **eigenvector** of \(\mathcal{P}_{G_l}\) (with eigenvalue either 0 or 1) and that eigenvectors corresponding to different subsystems are orthogonal.  
If additionally each subsystem satisfies its own goal (eigenvalue 1), then the collection is **perfectly aligned**.

### 5.4. Foam and the Total Functional

In the recursive definition of the total functional \(J^{(l)}\) ([gra_basics.md](gra_basics.md)), foam appears as a **penalty term**:

\[
J^{(l)}(\Psi^{(\mathbf{a})}) = \sum_{\mathbf{b}\prec\mathbf{a}} J^{(l-1)}(\Psi^{(\mathbf{b})}) \;+\; \Phi^{(l)}(\Psi^{(\mathbf{a})}, G_l^{(\mathbf{a})}) .
\]

Thus, optimizing the total functional drives foam to zero at every level.

---

## 6. Computing Foam in Practice

Explicitly constructing projectors \(\mathcal{P}_{G_l}\) is often infeasible for high‑dimensional spaces.  
In practice, we use **surrogates**:

### 6.1. Loss Functions

For many goals, we have a differentiable **loss function** \(\mathcal{L}_G(\psi)\) that is zero iff \(\psi\) satisfies the goal.  
Then we can approximate the projector’s action by:

\[
\mathcal{P}_G |\psi\rangle \approx |\psi\rangle - \eta \nabla_\psi \mathcal{L}_G(\psi) .
\]

This is effectively one step of gradient descent on the loss.

### 6.2. Embeddings and Contrastive Learning

The inner product \(\langle \psi | \mathcal{P}_G | \phi \rangle\) can be estimated by:

- Training an encoder \(f\) that maps states to embeddings.
- Using a **contrastive loss** to make embeddings of goal‑satisfying states similar and embeddings of non‑satisfying states orthogonal.
- Then \(\langle \psi | \mathcal{P}_G | \phi \rangle \approx f(\psi)^\top f(\phi)\) (if embeddings are normalized).

### 6.3. Monte Carlo Sampling

For very large spaces, we can approximate sums over pairs by random sampling:

\[
\Phi^{(l)} \approx \frac{N_l(N_l-1)}{2} \cdot \mathbb{E}_{\mathbf{a}\neq\mathbf{b}} \bigl[ |\langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle|^2 \bigr],
\]

where the expectation is taken over randomly chosen pairs.

### 6.4. Gradient of Foam

To minimize foam, we need gradients with respect to each \(\Psi^{(\mathbf{a})}\).  
From the definition:

\[
\frac{\partial \Phi^{(l)}}{\partial \Psi^{(\mathbf{a})}} = 2 \sum_{\mathbf{b}\neq\mathbf{a}} \langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle \cdot \mathcal{P}_{G_l} |\Psi^{(\mathbf{b})}\rangle .
\]

This can be computed if we have a differentiable model for \(\mathcal{P}_{G_l}\) (e.g., via the loss approximation).

---

## 7. Examples of Projectors in Physical AI

### 7.1. Camera Calibration (G₀)

Let \(\mathcal{H}\) be the space of images \(\mathbb{R}^{640\times480\times3}\).  
The calibrated subspace is the set of images with known intrinsic parameters.  
A practical projector: apply calibration algorithm to estimate parameters, then reproject to “calibrated” image.  
Loss function: reprojection error.

### 7.2. Time Synchronization (G₁)

For two cameras A and B, the goal subspace requires that timestamps \(t_A\) and \(t_B\) satisfy \(|t_A - t_B| < \delta\).  
Projector: average the two images if they are within δ, otherwise flag inconsistency.  
Foam: difference between actual timestamps and required synchronization.

### 7.3. Collision‑Free Trajectory (G₂)

Let \(\tau\) be a trajectory. The collision‑free subspace is defined by the world model \(M\): \(M(\tau) = \text{``no collision''}\).  
Projector: project onto the set of trajectories that satisfy this predicate – this is essentially a **constraint satisfaction** problem.  
In practice, we use a **collision checker** that returns a binary flag, and we can define a soft penalty (e.g., distance to obstacles) as a differentiable surrogate.

### 7.4. Ethical Constraint (G₄)

The “do no harm” subspace is the set of actions that do not cause harm.  
This is often implemented as a **filter**: any action that violates the constraint is projected to the nearest harmless action (e.g., by scaling down forces).  
The projector is then a **hard‑coded** operation, not learned.

---

## 8. Connection to the Zeroing Theorem

The zeroing theorem ([theorems.md](theorems.md)) states that if all projectors commute and the hierarchy consistency holds, there exists a state \(\Psi^*\) with zero foam at all levels.  
In terms of projectors, this state is a **common eigenvector** of all projectors with eigenvalue 1.

**Proof idea** (constructive):  
Build \(\Psi^*\) recursively as a tensor product of lower‑level eigenvectors.  
Because of commutativity, these eigenvectors can be chosen simultaneously for all projectors at a given level.  
The off‑diagonal elements vanish because the state is a product state in the eigenbasis.

---

## 9. Implementation Notes

### 9.1. Differentiable Projectors

To integrate with gradient‑based optimization, we need differentiable approximations of projectors.  
Common approaches:

- **Soft constraints**: replace hard projections with penalties (e.g., quadratic barrier functions).
- **Learned projectors**: train a neural network to map any state to the closest goal‑satisfying state.
- **Adversarial methods**: use GAN‑like discriminators to determine if a state satisfies the goal.

### 9.2. Scalability

For large numbers of subsystems, computing all pairwise inner products is \(O(N^2)\).  
Use mini‑batches, random sampling, or Nyström approximations to reduce cost.

### 9.3. Handling Non‑Commutativity

If goals are found to be non‑commutative, we have two options:

1. **Redesign** goals to be compatible (e.g., relax one of them).
2. Accept that full zeroing is impossible and instead seek a state that **minimizes** a weighted sum of foams (Pareto optimum).

The GRA framework primarily focuses on the case where full zeroing is achievable, as this yields the cleanest theoretical guarantees.

---

## 10. Summary

- **Projectors** are the mathematical representation of goals: they project onto the subspace of goal‑satisfying states.
- **Foam** measures inconsistency between subsystems by summing squared off‑diagonal elements of projectors.
- **Commutativity** and **hierarchy consistency** are necessary for the existence of a fully zeroed state.
- In practice, projectors can be approximated by loss functions, constraints, or learned models.
- The recursive zeroing algorithm uses gradients of foam to drive the system toward consistency.

Understanding projectors and foam is essential for implementing GRA in any real‑world system, especially in physical AI where multiple levels of goals must be aligned.

---

## Further Reading

- [gra_basics.md](gra_basics.md) – introduction to GRA concepts.
- [multiverse.md](multiverse.md) – multi‑indices and state spaces.
- [theorems.md](theorems.md) – the zeroing theorem and its proof.
- [layers.md](layers.md) – the hierarchy of goals G₀…Gₖ.
- [algorithm.md](../architecture/algorithm.md) – the recursive zeroing algorithm.

---

*“The goal is to make the invisible visible.”* – Paul Klee  
In GRA, we make the **inconsistencies** visible as foam, so they can be eliminated.
```