# Numerical Optimization Coursework

Graduate engineering optimization coursework exercises implemented in standalone Python.

## Overview

`constrained_newton_demo.py` solves a constrained quadratic program using an exterior quadratic penalty method across increasing penalty weights ($\rho \in \{1, 10, 100, 1000\}$), comparing:

- **Steepest Descent** with Armijo backtracking line search
- **Damped Newton's Method** using exact analytical gradients and Hessians

## Run

```bash
python constrained_newton_demo.py
```

Example output:

```text
rho       | SD iters | SD (x1, x2)         | Newton iters | Newton (x1, x2)
--------------------------------------------------------------------------
      1.0 |       67 | ( 1.6000,  1.3000) |            2 | ( 1.6000,  1.3000)
     10.0 |       80 | ( 1.3750,  1.1875) |            3 | ( 1.3750,  1.1875)
    100.0 |       80 | ( 1.3185,  1.1878) |            3 | ( 1.3377,  1.1689)
   1000.0 |       80 | ( 1.4602,  1.0409) |            4 | ( 1.3338,  1.1669)
```
