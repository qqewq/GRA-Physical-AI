```markdown
# Recursive Zeroing Algorithm: Finding the Coherent State

[< back to Documentation](../README.md) | [previous: projections.md](projections.md) | [next: examples/](../examples/)

The **Multiverse Zeroing Theorem** ([theorems.md](theorems.md)) guarantees the existence of a fully coherent state \(\Psi^*\) with zero foam at all levels.  
But how do we **find** that state in practice?

This document presents the **recursive zeroing algorithm** – a practical, scalable method to drive any hierarchical system toward consistency.  
We give both the high‑level recursive formulation and a parallelized version suitable for large‑scale physical AI systems.

---

## 1. The Challenge

A physical AI system (robot) has:

- Millions of parameters (neural network weights, control gains, calibration values).
- Multiple levels of goals (\(G_0\) to \(G_K\)), each with its own foam term.
- Complex interactions between levels: changing something at level \(l\) affects foam at levels \(l-1\) and \(l+1\).

A naive approach – optimizing all parameters simultaneously with a single loss – fails because:

- The loss landscape is extremely high‑dimensional and non‑convex.
- Foam terms involve **pairwise** interactions between many subsystems, making gradients expensive.
- The recursive structure of goals means that lower‑level errors propagate upward.

The **recursive zeroing algorithm** exploits the hierarchical structure to break the problem into manageable pieces.

---

## 2. High‑Level Idea

The algorithm follows the **recursive definition** of the total functional \(J_{\text{total}}\):

- **Bottom‑up pass**: For each level from \(l=0\) to \(K\), ensure that all subsystems at that level satisfy their local goals (minimize \(J_{\text{local}}\)).
- **Top‑down pass**: For each level from \(l=K\) down to \(1\), adjust subsystems to reduce foam at that level, using gradients that incorporate higher‑level information.

This is analogous to **backpropagation** in neural networks, but here the “error signal” is the foam, and the “layers” are the goal levels.

---

## 3. The Recursive Algorithm

### 3.1. Core Recursive Function

```python
def zero_level(l, Psi, G_l, tol=1e-3, max_iters=100):
    """
    Recursively zero foam at level l and below.
    
    Args:
        l: current level (0..K)
        Psi: dictionary mapping multi‑indices to state vectors
        G_l: goal at level l (provides projector P_G and local cost J_local)
        tol: tolerance for foam at this level
        max_iters: maximum iterations for foam reduction
    
    Returns:
        Updated Psi with zero foam at levels ≤ l
    """
    
    # Base case: level 0 – just optimize local goals
    if l == 0:
        for a in indices_at_level(0):
            # Gradient descent on local cost
            for _ in range(local_iters):
                grad = gradient(J_local(Psi[a], G_0[a]))
                Psi[a] -= learning_rate_0 * grad
        return Psi
    
    # Recursive case: first zero all lower levels
    subsystems = decompose(Psi, l)  # get all subsystems at level l-1 contained in each level‑l system
    
    # For each subsystem at level l-1, zero it recursively
    for a in indices_at_level(l):
        for b in subsystems[a]:  # b is a multi‑index of length l-1
            Psi = zero_level(l-1, Psi, G_{l-1}[b])
    
    # Now we have lower levels zeroed – reduce foam at level l
    Psi_l = collect_level(l, Psi)  # extract all states with dim = l
    
    for iteration in range(max_iters):
        # Compute current foam at level l
        Phi = compute_foam(Psi_l, G_l)
        
        if Phi < tol:
            break
        
        # For each subsystem at level l, compute gradient of foam w.r.t. its state
        for a in indices_at_level(l):
            grad = foam_gradient(a, Psi_l, G_l)
            Psi[a] -= learning_rate_l * grad
    
    return Psi
```

### 3.2. Main Entry Point

```python
def zero_multiverse(Psi_init, goals, K, tolerances, learning_rates):
    """
    Recursively zero all levels from K down to 0.
    
    Args:
        Psi_init: initial state (dictionary multi‑index → vector)
        goals: list of goals G_0..G_K
        K: top level
        tolerances: list of tolerances ε_l for each level
        learning_rates: list of learning rates η_l for each level
    
    Returns:
        Fully zeroed state Psi_star
    """
    Psi = Psi_init.copy()
    
    # Start recursion from the top level
    Psi = zero_level(K, Psi, goals[K], tolerances[K])
    
    return Psi
```

---

## 4. Key Subroutines

### 4.1. Computing Foam

```python
def compute_foam(Psi_l, G_l):
    """
    Compute Φ⁽ˡ⁾(Ψ⁽ˡ⁾, G_l).
    
    Args:
        Psi_l: dict {a: state} for all a with dim(a)=l
        G_l: goal at level l (provides projector P_G)
    
    Returns:
        scalar foam value
    """
    foam = 0.0
    indices = list(Psi_l.keys())
    
    # Sum over all distinct pairs
    for i, a in enumerate(indices):
        for b in indices[i+1:]:
            # Compute <Ψ^(a)| P_G |Ψ^(b)>
            overlap = inner_product(Psi_l[a], apply_projector(Psi_l[b], G_l))
            foam += abs(overlap)**2
    
    return foam
```

### 4.2. Gradient of Foam

From [projections.md](projections.md):

\[
\frac{\partial \Phi^{(l)}}{\partial \Psi^{(\mathbf{a})}} = 2 \sum_{\mathbf{b}\neq\mathbf{a}} \langle \Psi^{(\mathbf{a})} | \mathcal{P}_{G_l} | \Psi^{(\mathbf{b})} \rangle \cdot \mathcal{P}_{G_l} |\Psi^{(\mathbf{b})}\rangle .
\]

Implementation:

```python
def foam_gradient(a, Psi_l, G_l):
    """
    Compute gradient of Φ⁽ˡ⁾ w.r.t. Ψ^(a).
    
    Args:
        a: multi‑index of length l
        Psi_l: dict of all states at level l
        G_l: goal at level l
    
    Returns:
        gradient vector (same shape as Psi_l[a])
    """
    grad = zero_like(Psi_l[a])
    
    for b in Psi_l.keys():
        if b == a:
            continue
        # Compute overlap
        proj_b = apply_projector(Psi_l[b], G_l)
        overlap = inner_product(Psi_l[a], proj_b)
        # Add contribution
        grad += 2 * overlap * proj_b
    
    return grad
```

### 4.3. Applying a Projector

```python
def apply_projector(state, goal):
    """
    Apply projector P_G to a state.
    
    In practice, this might be:
    - A neural network that maps state to goal‑satisfying state
    - A gradient step on a loss function
    - A hard constraint projection (e.g., clipping)
    """
    if hasattr(goal, 'project'):
        # If goal provides a direct projection method
        return goal.project(state)
    else:
        # Fallback: one step of gradient descent on goal loss
        loss = goal.loss(state)
        grad = gradient(loss, state)
        return state - 0.1 * grad  # small step toward goal
```

### 4.4. Inner Product

For neural network states (weight vectors), inner product is the usual Euclidean dot product.  
For more complex state representations (e.g., images, point clouds), we may need a **kernel** or **embedding**:

```python
def inner_product(state1, state2):
    if isinstance(state1, np.ndarray):
        return np.dot(state1.flatten(), state2.flatten())
    else:
        # Use a predefined kernel
        return kernel(state1, state2)
```

---

## 5. Parallel Version

The recursive algorithm is inherently sequential: we must finish lower levels before adjusting higher ones.  
However, within a level, updates can be **parallelized** across subsystems.

### 5.1. Parallel Foam Gradient

```python
def parallel_foam_gradient(Psi_l, G_l):
    """
    Compute gradients for all subsystems at level l in parallel.
    
    Returns:
        dict {a: grad_a} for all a
    """
    indices = list(Psi_l.keys())
    N = len(indices)
    
    # Compute all projected states first (can be done in parallel)
    proj = {a: apply_projector(Psi_l[a], G_l) for a in indices}
    
    # Compute all inner products (matrix of overlaps)
    overlaps = np.zeros((N, N))
    for i, a in enumerate(indices):
        for j, b in enumerate(indices):
            if i != j:
                overlaps[i,j] = inner_product(Psi_l[a], proj[b])
    
    # For each a, sum contributions from all b ≠ a
    grads = {}
    for i, a in enumerate(indices):
        grad = zero_like(Psi_l[a])
        for j, b in enumerate(indices):
            if i != j:
                grad += 2 * overlaps[i,j] * proj[b]
        grads[a] = grad
    
    return grads
```

### 5.2. Parallel Update Rule

From [multiverse.md](multiverse.md), the full update for a subsystem at level \(l\) includes gradients from **higher** levels too:

\[
\Psi^{(\mathbf{a})}(t+1) = \Psi^{(\mathbf{a})}(t) - \eta \left[ \Lambda_l \frac{\partial \Phi^{(l)}}{\partial \Psi^{(\mathbf{a})}} + \sum_{\mathbf{b} \succ \mathbf{a}} \Lambda_{l+1} \frac{\partial \Phi^{(l+1)}}{\partial \Psi^{(\mathbf{a})}} \right]
\]

This can be implemented as:

```python
def update_all_levels(Psi, goals, Lambdas, learning_rate):
    """
    One parallel update for all subsystems at all levels.
    """
    K = len(goals) - 1
    
    # Compute gradients for each level
    grads = {l: parallel_foam_gradient(collect_level(l, Psi), goals[l]) 
             for l in range(K+1)}
    
    # For each subsystem, combine gradients from its own level and all higher levels
    new_Psi = Psi.copy()
    for a in Psi.keys():
        l = len(a) - 1  # level of this subsystem
        grad = Lambdas[l] * grads[l][a]
        
        # Add contributions from all higher levels that contain this subsystem
        for m in range(l+1, K+1):
            for b in indices_containing(a, m):  # find all level‑m indices that have a as prefix
                grad += Lambdas[m] * grads[m][b]  # contribution through chain rule
        
        new_Psi[a] = Psi[a] - learning_rate * grad
    
    return new_Psi
```

---

## 6. Practical Considerations

### 6.1. Convergence Criteria

We don't need to achieve exactly zero foam – a small tolerance is enough:

\[
\Phi^{(l)} < \varepsilon_l \quad \forall l
\]

Typical values: \(\varepsilon_0 = 10^{-6}\) (hardware precision), \(\varepsilon_K = 0.01\) (ethical constraints can be softer).

### 6.2. Adaptive Learning Rates

Higher levels often need smaller learning rates to avoid destabilizing lower levels.  
Use \(\eta_l = \eta_0 \cdot \beta^l\) with \(\beta < 1\).

### 6.3. Warm‑up Phase

Start by optimizing only level 0 (local goals) for several iterations. Then gradually introduce higher levels. This prevents the system from chasing inconsistent high‑level goals before the basics are right.

### 6.4. Handling Non‑Differentiable Goals

Some goals (e.g., hard safety constraints) are not differentiable.  
Use **augmented Lagrangian** methods: treat them as constraints and introduce Lagrange multipliers.

### 6.5. Scalability

For large \(N\) (number of subsystems at a level), computing all \(O(N^2)\) overlaps is expensive.  
Use:

- **Random sampling**: estimate foam from a random subset of pairs.
- **Locality**: only pairs that are “close” in the hierarchy or physically interact need to be considered.
- **Nyström approximation**: for kernel‑based inner products.

---

## 7. Example: Two‑Level Robot

Let's walk through a concrete example with a simple robot:

- **Level 0**: two motors (left, right). State = target velocities. Goal: achieve commanded velocity within 5%.
- **Level 1**: planner that outputs velocities to reach a goal. Goal: planned path is collision‑free.

**Initialization**: random motor gains, random planner weights.

**Recursive zeroing**:

1. `zero_level(1, ...)` calls `zero_level(0, ...)` for each motor.
2. Level 0 optimizes: motors learn to track commands accurately (reduce local cost).
3. Back to level 1: compute foam = difference between planner's desired velocities and what motors can actually achieve.
4. Update planner to request feasible velocities, and update motors to respond faster.
5. Repeat until foam < ε.

After convergence, the robot moves smoothly – planner and motors are perfectly coordinated.

---

## 8. Relation to Other Algorithms

- **Backpropagation**: Our algorithm is a form of **recursive gradient descent** through a hierarchical loss.  
  The key difference: the loss at level \(l\) is not a simple function of the output, but a pairwise interaction term (foam).
- **Alternating optimization**: We alternate between optimizing lower levels and then higher levels, similar to coordinate descent.
- **Multigrid methods**: Like multigrid, we solve coarse problems (high level) and then refine fine details (low level).

---

## 9. Summary

The **recursive zeroing algorithm**:

- Exploits the hierarchical structure of goals.
- Guarantees convergence to a zero‑foam state under mild conditions (convexity of local costs, differentiability of projectors).
- Can be parallelized across subsystems within a level.
- Scales polynomially with the number of subsystems.

For physical AI, this algorithm provides a practical way to train robots that are **internally consistent** – where every level, from raw motors to high‑level ethics, works in harmony.

---

## Further Reading

- [gra_basics.md](../theory/gra_basics.md) – mathematical foundations.
- [multiverse.md](../theory/multiverse.md) – multi‑indices and state spaces.
- [theorems.md](../theory/theorems.md) – the zeroing theorem.
- [layers.md](layers.md) – hierarchy of goals G₀…Gₖ.
- [projections.md](projections.md) – projectors and foam.
- [examples/](../examples/) – full use‑cases.

---

*“Recursion – see ‘Recursion’.”* – Anonymous  
In GRA, recursion is the key to unlocking consistency across infinite levels.
```