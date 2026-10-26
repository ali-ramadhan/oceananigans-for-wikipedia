# Two-dimensional Taylor–Green vortex

`taylor_green_2d.jl` simulates the decaying two-dimensional Taylor–Green vortex with
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), compares it with the exact solution, and animates its
vorticity, streamlines and fluid particles for the Wikipedia article
[Taylor–Green vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex).

## Problem

The simulation solves the incompressible Navier–Stokes equations with constant viscosity $\nu$,

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0,
```

for the velocity $\mathbf{u} = (u, v)$ and the pressure $p$ (divided by the density), in a doubly periodic square of
side $2\pi/k$. The flow starts as a 2 × 2 array of counter-rotating vortices, each of size $\pi/k$, with velocity
amplitude $U_0$:

```math
u = U_0 \sin(kx) \cos(ky), \qquad v = -U_0 \cos(kx) \sin(ky).
```

The pressure gradient exactly balances the advection term, so viscosity weakens the vortices without changing their
shape. The exact solution is

```math
\begin{aligned}
u &= U_0 \sin(kx) \cos(ky) \, F(t), & v &= -U_0 \cos(kx) \sin(ky) \, F(t), \\
p &= \frac{U_0^2}{4} \left[ \cos(2kx) + \cos(2ky) \right] F(t)^2, & F(t) &= e^{-2 \nu k^2 t},
\end{aligned}
```

and its vorticity is $\omega = \partial_x v - \partial_y u = 2 k U_0 \sin(kx) \sin(ky) \, F(t)$.

## Set-up

- $U_0 = k = 1$ and $\nu = 0.01$, so the Reynolds number is $\mathrm{Re} = U_0 / (k \nu) = 100$.
- The run ends at $t = 100 / (k U_0)$, when the kinetic energy has decayed to $e^{-4} \approx 2\%$ of its initial value.
- Oceananigans 0.114.1 `NonhydrostaticModel` with its defaults: finite volumes on a staggered grid, second-order
  centered advection, third-order Runge–Kutta time stepping and an FFT-based pressure solver.
- 256 × 256 grid cells, so $k \Delta x = 2\pi / 256$.
- Adaptive time step: advective CFL number at most 0.5, and $\nu \Delta t / \Delta x^2$ at most 0.2.
- 576 Lagrangian particles, starting on a uniform 24 × 24 lattice, are carried by the flow for the animation.

## Accuracy

At $t = 100 / (k U_0)$ the maximum error relative to the current amplitude is $7.5 \times 10^{-5}$ in vorticity and
$1.1 \times 10^{-4}$ in pressure. Throughout the run the kinetic energy stays within a relative $2 \times 10^{-4}$ of
the exact $\tfrac{1}{4} U_0^2 e^{-4 \nu k^2 t}$. These are second-order truncation errors, mostly from the discrete
Laplacian: it makes the vortices decay at the slightly slower rate $2 \nu k^2 \left[ 1 - (k \Delta x)^2 / 12 \right]$,
which leaves their amplitude too large by $(k \Delta x)^2 / 6 \approx 10^{-4}$ at the end.
