"""Constrained quadratic optimization via exterior penalty, steepest descent, and damped Newton."""

from __future__ import annotations

import math


def objective_with_penalty(x1: float, x2: float, rho: float) -> float:
    """Minimize (x1 - 2)^2 + 2*(x2 - 1.5)^2 + rho * max(0, x1 + x2 - 2.5)^2."""
    base = (x1 - 2.0) ** 2 + 2.0 * (x2 - 1.5) ** 2
    violation = max(0.0, x1 + x2 - 2.5)
    return base + rho * (violation ** 2)


def grad_and_hessian(x1: float, x2: float, rho: float) -> tuple[tuple[float, float], tuple[float, float, float]]:
    """Compute exact gradient and Hessian (h11, h12, h22) for the penalty objective."""
    g1 = 2.0 * (x1 - 2.0)
    g2 = 4.0 * (x2 - 1.5)
    h11, h12, h22 = 2.0, 0.0, 4.0
    violation = x1 + x2 - 2.5
    if violation > 0.0:
        g1 += 2.0 * rho * violation
        g2 += 2.0 * rho * violation
        h11 += 2.0 * rho
        h12 += 2.0 * rho
        h22 += 2.0 * rho
    return (g1, g2), (h11, h12, h22)


def solve_steepest_descent(rho: float, max_iter: int = 80, tol: float = 1e-6) -> tuple[float, float, int]:
    """Armijo backtracking steepest descent."""
    x1, x2 = 0.0, 0.0
    for k in range(1, max_iter + 1):
        (g1, g2), _ = grad_and_hessian(x1, x2, rho)
        gnorm = math.hypot(g1, g2)
        if gnorm < tol:
            return x1, x2, k - 1
        alpha = 1.0
        f0 = objective_with_penalty(x1, x2, rho)
        while alpha > 1e-8:
            nx1, nx2 = x1 - alpha * g1, x2 - alpha * g2
            if objective_with_penalty(nx1, nx2, rho) <= f0 - 1e-4 * alpha * (g1 * g1 + g2 * g2):
                x1, x2 = nx1, nx2
                break
            alpha *= 0.5
    return x1, x2, max_iter


def solve_damped_newton(rho: float, max_iter: int = 30, tol: float = 1e-8) -> tuple[float, float, int]:
    """2x2 analytical Newton step with backtracking line search."""
    x1, x2 = 0.0, 0.0
    for k in range(1, max_iter + 1):
        (g1, g2), (h11, h12, h22) = grad_and_hessian(x1, x2, rho)
        if math.hypot(g1, g2) < tol:
            return x1, x2, k - 1
        det = h11 * h22 - h12 * h12
        d1 = -(h22 * g1 - h12 * g2) / det
        d2 = -(-h12 * g1 + h11 * g2) / det
        slope = g1 * d1 + g2 * d2
        alpha = 1.0
        f0 = objective_with_penalty(x1, x2, rho)
        while alpha > 1e-8:
            nx1, nx2 = x1 + alpha * d1, x2 + alpha * d2
            if objective_with_penalty(nx1, nx2, rho) <= f0 + 1e-4 * alpha * slope:
                x1, x2 = nx1, nx2
                break
            alpha *= 0.5
    return x1, x2, max_iter


def main() -> None:
    print("rho       | SD iters | SD (x1, x2)         | Newton iters | Newton (x1, x2)")
    print("-" * 74)
    for rho in (1.0, 10.0, 100.0, 1000.0):
        sd1, sd2, sdk = solve_steepest_descent(rho)
        nw1, nw2, nwk = solve_damped_newton(rho)
        print(f"{rho:9.1f} | {sdk:8d} | ({sd1:7.4f}, {sd2:7.4f}) | {nwk:12d} | ({nw1:7.4f}, {nw2:7.4f})")


if __name__ == "__main__":
    main()
