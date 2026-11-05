# Three-dimensional Taylor–Green vortex

Simulations of the three-dimensional Taylor–Green vortex with
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), for the Wikipedia article
[Taylor–Green vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex):

- `simulate.jl` simulates the vortex at Re = 1600 in three dimensions and in two, and saves everything the animation
  scripts read.
- `animate_3d.jl` animates the three-dimensional flow in five styles, each to its own video.
- `animate_2d_vs_3d.jl` animates the two- and three-dimensional vortex side by side.
- `taylor_green.jl` holds the code shared by the scripts.
- `video_encoding.jl` records the animations and encodes them for Wikimedia Commons.

## Problem

The simulations solve the incompressible Navier–Stokes equations with constant viscosity $\nu$ in the triply periodic
cube $-\pi L \le x, y, z < \pi L$. The flow starts from the standard benchmark initial condition,

```math
u = V_0 \sin(x/L) \cos(y/L) \cos(z/L), \qquad v = -V_0 \cos(x/L) \sin(y/L) \cos(z/L), \qquad w = 0,
```

at Reynolds number $\mathrm{Re} = V_0 L / \nu = 1600$. Each horizontal plane starts as a two-dimensional Taylor–Green
vortex whose strength varies with height as $\cos(z/L)$.

The two-dimensional comparison simulates the plane $z = 0$ on its own, where this initial condition is an exact solution
that decays as $e^{-2 \nu t / L^2}$.

## Set-up

- $V_0 = L = 1$ and $\nu = 1/1600$.
- Oceananigans 0.116.1 `NonhydrostaticModel`: fifth-order WENO advection, third-order Runge–Kutta time stepping and an
  FFT-based pressure solver, in double precision on the GPU.
- 512³ cells in three dimensions and 512² cells in two.
- Adaptive time step: advective CFL number at most 0.6.
- The runs end at $t = 60 L/V_0$, with output every $0.05 L/V_0$:
  - the three-dimensional velocity up to $t = 20 L/V_0$, in single precision, which takes 705 GB;
  - the vorticity in the plane $z = \pi L/4$ and the mean squares of the velocity and vorticity components;
  - the two-dimensional vorticity and its mean square.

## Animations

`animate_3d.jl` draws the three-dimensional flow up to $t = 20 L/V_0$, from a fixed camera:

- `strongest`: the surface enclosing the 1% of the volume where $|\boldsymbol{\omega}|$ is largest, colored by the
  direction of the vortex lines, $\omega_z / |\boldsymbol{\omega}|$.
- `level`: the surface $|\boldsymbol{\omega}| = 7\, V_0/L$, colored by $\omega_z / |\boldsymbol{\omega}|$.
- `q`: the surface where the Q-criterion, $Q = -\tfrac{1}{2} \partial_j u_i\, \partial_i u_j$, equals
  $10\, V_0^2/L^2$, colored by $|\boldsymbol{\omega}|$.
- `volume`: a volume rendering of $|\boldsymbol{\omega}|$.
- `annotated`: the `strongest` surface, paused at $t$ = 0, 3, 5, 7, 9 and $20\, L/V_0$ while a label and arrows explain
  each stage of the transition to turbulence.

`animate_2d_vs_3d.jl` shows the two-dimensional vorticity beside the vertical vorticity $\omega_z$ in the plane
$z = \pi L/4$ of the three-dimensional flow, up to $t = 60 L/V_0$. That plane starts as the two-dimensional vortex at
$\cos(\pi/4) \approx 71\%$ of its strength. The plane $z = 0$ would show little, because the flow's symmetry keeps it
calm.

## Accuracy

The three-dimensional run is compared with the benchmark's pseudo-spectral reference solution on 512³ modes:

- The kinetic-energy dissipation rate $-\mathrm{d}E/\mathrm{d}t$ peaks at $0.01273\, V_0^3/L$ at $t = 8.95\, L/V_0$. The
  reference peaks at $0.01286\, V_0^3/L$ at $t = 8.97\, L/V_0$, so the run is 1.0% low.
- Up to $t = 20 L/V_0$, where the reference ends, the dissipation rate differs from the reference by at most 1.8% of
  the reference's peak.
- The mean-square vorticity grows 26.6-fold by the peak, against 27.4-fold in the reference.
- The viscous dissipation $\nu \langle |\boldsymbol{\omega}|^2 \rangle$ peaks at $0.01248\, V_0^3/L$, so WENO's numerical
  dissipation is about 2% of the total near the peak.

In two dimensions, the mean-square vorticity at $t = 60 L/V_0$ is 0.860710 of its initial value, against the exact
$e^{-4 \nu t / L^2} = 0.860708$, and its relative error stays below $2 \times 10^{-6}$ throughout.

The flow is symmetric under the mirrors $x \to -x$ and $y \to -y$. On the plane $z = \pi L/4$, the antisymmetric part of
$\omega_z$ stays below $2 \times 10^{-9}$ of the whole up to $t = 60 L/V_0$, the size of the single-precision rounding
in the saved output.
