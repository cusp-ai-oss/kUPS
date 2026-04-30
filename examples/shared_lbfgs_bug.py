# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproduction: shared LBFGS across independent systems degrades convergence.

This script demonstrates that when two independent quadratic systems with
different curvatures share a single LBFGS optimizer (as happens in kUPS's
batched relaxation), the inverse Hessian approximation is corrupted and
convergence degrades catastrophically.

No kUPS installation or ML model needed — pure NumPy.

Usage:
    python shared_lbfgs_bug.py
"""

import numpy as np


def lbfgs_two_loop(grad, s_list, y_list, rho_list, H0, memory, iteration):
    """Standard L-BFGS two-loop recursion (Nocedal Algorithm 7.4)."""
    loopmax = min(memory, iteration)
    a = np.empty(loopmax)
    q = grad.copy()
    for i in range(loopmax - 1, -1, -1):
        a[i] = rho_list[i] * np.dot(s_list[i], q)
        q -= a[i] * y_list[i]
    z = H0 * q
    for i in range(loopmax):
        b = rho_list[i] * np.dot(y_list[i], z)
        z += s_list[i] * (a[i] - b)
    return z


def run_independent_lbfgs(x0, grad_fn, steps=50, alpha=70.0, memory=100,
                          maxstep=0.2):
    """Run LBFGS on a single system (ASE-equivalent behavior)."""
    H0 = 1.0 / alpha
    s_list, y_list, rho_list = [], [], []
    x, r0, g0 = x0.copy(), None, None
    history = []
    for step in range(steps):
        g = grad_fn(x)
        history.append(float(np.max(np.abs(g))))
        if step > 0:
            s, y = x - r0, g - g0
            dot = np.dot(y, s)
            if abs(dot) > 1e-30:
                s_list.append(s)
                y_list.append(y)
                rho_list.append(1.0 / dot)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
        d = lbfgs_two_loop(g, s_list, y_list, rho_list, H0, memory, step)
        dmax = np.max(np.abs(d))
        if dmax > maxstep:
            d *= maxstep / dmax
        r0, g0 = x.copy(), g.copy()
        x = x - d
    return history


def run_shared_lbfgs(x0_a, x0_b, grad_a, grad_b, steps=50, alpha=70.0,
                     memory=100, maxstep=0.2):
    """Run ONE shared LBFGS over concatenated [x_a, x_b].

    This mimics kUPS's current batched relaxation behavior: all systems'
    positions are concatenated and one optimizer state is shared.
    """
    H0 = 1.0 / alpha
    s_list, y_list, rho_list = [], [], []
    n_a = len(x0_a)
    x = np.concatenate([x0_a, x0_b])
    r0, g0 = None, None
    hist_a, hist_b = [], []
    for step in range(steps):
        g = np.concatenate([grad_a(x[:n_a]), grad_b(x[n_a:])])
        hist_a.append(float(np.max(np.abs(g[:n_a]))))
        hist_b.append(float(np.max(np.abs(g[n_a:]))))
        if step > 0:
            s, y = x - r0, g - g0
            dot = np.dot(y, s)
            if abs(dot) > 1e-30:
                s_list.append(s)
                y_list.append(y)
                rho_list.append(1.0 / dot)
                if len(s_list) > memory:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
        d = lbfgs_two_loop(g, s_list, y_list, rho_list, H0, memory, step)
        dmax = np.max(np.abs(d))
        if dmax > maxstep:
            d *= maxstep / dmax
        r0, g0 = x.copy(), g.copy()
        x = x - d
    return hist_a, hist_b


def main():
    np.random.seed(42)
    dim = 81  # 27 atoms × 3 coordinates (like a small drug molecule)

    # System A: steep, well-conditioned (eigenvalues 10–100)
    Q_a = np.linalg.qr(np.random.randn(dim, dim))[0]
    H_a = Q_a @ np.diag(np.linspace(10, 100, dim)) @ Q_a.T

    # System B: shallow, ill-conditioned (eigenvalues 0.1–10)
    Q_b = np.linalg.qr(np.random.randn(dim, dim))[0]
    H_b = Q_b @ np.diag(np.linspace(0.1, 10, dim)) @ Q_b.T

    x0 = np.random.randn(dim) * 0.5

    # Independent LBFGS (correct behavior — each system gets its own state)
    indep_a = run_independent_lbfgs(x0, lambda x: H_a @ x, steps=50)
    indep_b = run_independent_lbfgs(x0, lambda x: H_b @ x, steps=50)

    # Shared LBFGS (kUPS batched behavior — one state for both)
    batch_a, batch_b = run_shared_lbfgs(
        x0, x0, lambda x: H_a @ x, lambda x: H_b @ x, steps=50
    )

    print("=" * 70)
    print("  Shared vs Independent LBFGS on independent quadratic systems")
    print("=" * 70)
    print(f"  System A: eigenvalues [10, 100], dim={dim}")
    print(f"  System B: eigenvalues [0.1, 10], dim={dim}")
    print(f"  alpha=70, memory=100, maxstep=0.2, steps=50\n")

    print(f"  {'Step':>4}  {'A indep':>10} {'A shared':>10} {'ratio':>10}"
          f"  {'B indep':>10} {'B shared':>10} {'ratio':>10}")
    print("  " + "-" * 66)
    for i in [0, 5, 10, 20, 30, 40, 49]:
        ra = batch_a[i] / indep_a[i] if indep_a[i] > 1e-30 else float("inf")
        rb = batch_b[i] / indep_b[i] if indep_b[i] > 1e-30 else float("inf")
        print(f"  {i:>4}  {indep_a[i]:>10.2e} {batch_a[i]:>10.2e} {ra:>9.1f}x"
              f"  {indep_b[i]:>10.2e} {batch_b[i]:>10.2e} {rb:>9.1f}x")

    print(f"\n  Final gmax after 50 steps:")
    ratio_a = batch_a[-1] / indep_a[-1] if indep_a[-1] > 1e-30 else float("inf")
    print(f"    System A: Independent={indep_a[-1]:.2e}  "
          f"Shared={batch_a[-1]:.2e}  ({ratio_a:.0f}x worse)")
    print(f"    System B: Independent={indep_b[-1]:.2e}  "
          f"Shared={batch_b[-1]:.2e}")
    print(f"\n  The shared LBFGS corrupts System A's convergence by mixing")
    print(f"  curvature information from the unrelated System B into the")
    print(f"  inverse Hessian approximation via the s·y dot products.")


if __name__ == "__main__":
    main()
