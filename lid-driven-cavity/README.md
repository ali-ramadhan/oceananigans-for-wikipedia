# Lid-driven cavity

Simulations of the two-dimensional lid-driven cavity with
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), for a proposed Wikipedia article:

- `simulate_evolution.jl` simulates the flow at Re = 5000 from rest until it is steady, and saves everything the next
  two scripts read.
- `animate_evolution.jl` animates the streamlines over the vorticity, for the lead of the article.
- `check_evolution.jl` checks the run against the same run on a coarser grid.
- `simulate_steady.jl` runs the flow at Re = 7500 until it is steady.
- `schematic.jl` draws `Lid-driven cavity schematic.svg` from that flow.
- `simulate_comparison.jl` simulates the flow at the other Reynolds numbers of the comparison from rest, until t = 300
  L/U or until it is steady.
- `animate_comparison.jl` animates six Reynolds numbers in one 4K video.
- `check_comparison.jl` prints how those runs end: their kinetic energy, and the main frequencies at Re = 10,000.
- `profiles.jl` draws `Lid-driven cavity velocity and vorticity profiles.png` from the steady flows at Re = 100, 1000
  and 5000.
- `scraping_flow.jl` draws `Taylor scraping flow in a right-angled corner.svg` from its closed-form solution, with no
  simulation.
- `moffatt_eddies.jl` draws `Moffatt eddies in a right-angled corner.svg` in the same way.
- `check_steady.jl` compares the steady flows at Re = 100, 1000, 5000 and 7500 with published values.
- `lid_driven_cavity.jl` holds the code shared by the scripts: the simulation, the saved runs and the fields derived
  from them.
- `plotting.jl` holds what the animations and the profiles figure share: the colors of the vorticity, the panel of
  vorticity under streamlines, and the videos.
- `published_results.jl` holds the published tables that `profiles.jl` and `check_steady.jl` compare with.
- `video_encoding.jl` records the animations and encodes them for Wikimedia Commons.
- `svg_drawing.jl` holds the colors and helpers of the scripts that write SVG files directly.

The evolution at Re = 5000 and the rest of the comparison use fifth-order WENO advection, and the schematic uses
second-order centered advection. `Manifest.toml` records the versions of Oceananigans and the other packages.

## Problem

The simulations solve the incompressible Navier–Stokes equations with constant viscosity $\nu$,

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u},
\qquad \nabla \cdot \mathbf{u} = 0,
```

in the square $0 < x, y < L$. The fluid starts at rest, and from $t = 0$ the top wall moves at speed $U$:

```math
(u, v) = (U, 0) \ \text{ on } y = L, \qquad (u, v) = (0, 0) \ \text{ on the other three walls},
```

at Reynolds number $\mathrm{Re} = UL/\nu$. The flow is restricted to two dimensions, whereas a real cavity flow at these
Reynolds numbers would be three-dimensional.

## Evolution to a steady flow at Re = 5000

The lid drags a thin vortex sheet along with it. At the downstream corner the wall strips the sheet off the lid, and it
rolls up into a starting vortex, which grows and reaches the middle of the cavity by about $t = 20\, L/U$. It becomes
the primary vortex, whose center keeps circling its final position about every $7.5\, L/U$ until about $t = 120\, L/U$.
A counter-rotating eddy forms on the right wall by about $t = 10\, L/U$ and slides down into the bottom right corner.
Others follow in the bottom left corner by about $t = 20\, L/U$ and in the top left corner by about $t = 25\, L/U$. The
flow stops changing visibly by about $t = 300\, L/U$.

### Set-up

- $U = L = 1$ and $\nu = 2 \times 10^{-4}$.
- Oceananigans `NonhydrostaticModel`: fifth-order WENO advection, third-order Runge–Kutta time stepping and an
  FFT-based pressure solver, in double precision on the GPU.
- 1024² uniform cells, so the cell Reynolds number $U \Delta x / \nu$ is 4.9.
- Fixed time step $\Delta t = 0.7\, \Delta x / U = 6.84 \times 10^{-4}\, L/U$, shortened slightly to land on the output
  times.
- The run stops when $u$ has changed by less than $10^{-6}\, U$ everywhere over the last $50\, L/U$. That happened at
  $t = 950\, L/U$, after 1,390,977 iterations and 3.2 hours on an RTX 4090.
- There are 2645 outputs, every $0.025\, L/U$ until $t = 40\, L/U$ and further apart after that, which take 34 GB: the
  vorticity on the cell corners in single precision, $u$ in double precision, and the mean squares of $u$, $v$ and
  $\omega$.

Run it with `julia --startup-file=no --project simulate_evolution.jl`. Skipping the startup file keeps Revise out of the
run, which would otherwise patch it if packages are updated in the meantime.

### Animation

`animate_evolution.jl` draws 1920 frames at 30 frames per second, a 64-second video.

- **Time:** the video runs at $0.75\, L/U$ per second until $t = 25\, L/U$. The pace then grows exponentially, with an
  e-folding of 5 seconds, and the clock shows the time throughout. At a constant pace the video would last 21 minutes.
  It ends by holding the last output, at $t = 947.7\, L/U$, for 3 seconds. On Wikipedia, `thumbtime=1:03` shows that
  frame until the reader presses play.
- **Pixels:** each pixel of the 1024 × 1024 panel is the average of the vorticity over one cell.
- **Colors:** cmocean's `curl`, following $\mathrm{asinh}(\omega L / 2U)$, which is roughly linear below $2U/L$ and
  logarithmic above it. They saturate at $\pm 30\, U/L$.
- **Streamlines:** contours of the stream function $\psi$, where $u = \partial\psi/\partial y$ and $v =
  -\partial\psi/\partial x$, at $\psi / UL = -0.1, -0.07, -0.04, -0.01, -10^{-4}, -2 \times 10^{-7}, 10^{-5}, 5 \times
  10^{-4}$ and $2 \times 10^{-3}$. The levels span several orders of magnitude to show each eddy, so the spacing of the
  lines doesn't compare speeds between the eddies.

### Accuracy

The final state agrees closely with the published values. The extrapolation in the second row is at second order, from
this run and the same run on 512² cells. The values of Bruneau and Saad are mirrored to a lid moving in $+x$.

| Primary vortex | $\psi$ ($UL$) | $\omega$ ($U/L$) | Center |
|---|---|---|---|
| This run, 1024² cells | −0.122148 | −1.93958 | (0.5146, 0.5352) |
| This run, extrapolated | −0.122228 | | |
| Erturk et al. (2005), extrapolated | −0.122233 | −1.940732 | |
| Erturk (2009), 1025² points | −0.121942 | −1.936291 | (0.5146, 0.5352) |
| Bruneau and Saad (2006), 2048² cells | −0.12197 | −1.9327 | (0.5147, 0.5352) |
| Ghia et al. (1982), 257² points | −0.118966 | −1.86016 | (0.5117, 0.5352) |

| Eddy | $\psi$ ($UL$) | Erturk (2009) | Center | Erturk (2009) | Along the bottom wall ($L$) | Ghia et al. | Along the side wall ($L$) | Ghia et al. |
|---|---|---|---|---|---|---|---|---|
| BL₁ | $1.3753 \times 10^{-3}$ | $1.3729 \times 10^{-3}$ | (0.0732, 0.1367) | (0.0732, 0.1367) | 0.3290 | 0.3184 | 0.2691 | 0.2643 |
| BR₁ | $3.0758 \times 10^{-3}$ | $3.0677 \times 10^{-3}$ | (0.8037, 0.0723) | (0.8057, 0.0732) | 0.3642 | 0.3565 | 0.4294 | 0.4180 |
| TL₁ | $1.4489 \times 10^{-3}$ | $1.4442 \times 10^{-3}$ | (0.0635, 0.9092) | (0.0635, 0.9092) | | | 0.2737 | 0.2693 |
| BR₂ | $-1.4448 \times 10^{-6}$ | $-1.4045 \times 10^{-6}$ | (0.9785, 0.0186) | (0.9785, 0.0186) | 0.0569 | 0.0528 | 0.0441 | 0.0417 |
| BL₂ | $-6.91 \times 10^{-8}$ | $-6.61 \times 10^{-8}$ | (0.0078, 0.0078) | (0.0078, 0.0078) | 0.0189 | 0.0156 | 0.0197 | 0.0163 |

The kinetic energy per unit mass is $0.047319\, U^2$, against $0.047290\, U^2$ from Bruneau and Saad on 2048² cells.

The evolution has converged too. The same run on 512² cells, with the time step doubled, gives:

| $t$ ($L/U$) | Kinetic energy, 1024² ($U^2$) | Difference on 512² | Primary vortex $\psi$, 1024² ($UL$) | Difference on 512² | Distance between the two centers ($L$) | Difference in $u$ (% of its rms) |
|---|---|---|---|---|---|---|
| 1 | 0.004468 | −0.8% | −0.02425 | −0.6% | 0.001 | 0.8 |
| 5 | 0.010389 | −0.6% | −0.06282 | −0.3% | 0.002 | 1.6 |
| 10 | 0.015846 | −0.6% | −0.08293 | −0.2% | 0.001 | 1.7 |
| 20 | 0.024797 | −0.5% | −0.10245 | −0.2% | 0.001 | 1.2 |
| 30 | 0.032097 | −0.5% | −0.10885 | −0.2% | 0.003 | 0.8 |
| 50 | 0.038488 | −0.4% | −0.10865 | −0.3% | 0.004 | 0.5 |
| 120 | 0.044086 | −0.5% | −0.11446 | −0.2% | 0.001 | 0.3 |
| 400 | 0.047206 | −0.5% | −0.12188 | −0.2% | 0.001 | 0.3 |

WENO-5 was also more accurate than second-order centered advection on the same grids, in steady runs with the same time
step. The references are Botella and Peyret (1998) at Re = 1000 and Bruneau and Saad (2006) at Re = 5000.

| Re | Cells | Error in $\psi$, WENO-5 | Error in $\psi$, centered |
|---|---|---|---|
| 1000 | 128² | 0.823% | 1.196% |
| 1000 | 256² | 0.194% | 0.302% |
| 1000 | 512² | 0.047% | 0.075% |
| 5000 | 256² | 0.846% | 1.469% |

Both converge at second order overall, since WENO-5 drops to lower order next to the walls.

## Six Reynolds numbers

`animate_comparison.jl` shows the flow from rest at Re = 100, 1000, 5000, 10,000, 22,000 and 100,000 in one 4K video, on
one clock, until $t = 300\, L/U$. At Re = 100 and 1000 the flow soon becomes steady, and at Re = 5000 it takes longer
and grows more eddies in the corners. Re = 10,000 is above the first Hopf bifurcation, near Re = 8000, so its flow keeps
oscillating. At Re = 22,000 and 100,000 the sheet shed from the lid breaks up into vortices that keep moving around the
cavity.

### Set-up

`simulate_comparison.jl` runs each panel except Re = 5000, which is the run described above, with the same model and
schemes:

| Re | Cells | Cell Reynolds number | $\Delta t$ ($L/U$) | Ends at $t$ ($L/U$) | Iterations | Time on an RTX 4090 | Output |
|---|---|---|---|---|---|---|---|
| 100 | 512² | 0.2 | $9.54 \times 10^{-5}$ | 40 | 420,800 | 21 min | 4.9 GB |
| 1000 | 512² | 2.0 | $9.54 \times 10^{-4}$ | 160 | 169,500 | 10 min | 9.4 GB |
| 5000 | 1024² | 4.9 | $6.84 \times 10^{-4}$ | 947.7 | 1,390,977 | 3.2 h | 34 GB |
| 10,000 | 1024² | 9.8 | $6.84 \times 10^{-4}$ | 300 | 441,500 | 1.1 h | 54 GB |
| 22,000 | 2048² | 10.7 | $3.42 \times 10^{-4}$ | 300 | 880,500 | 8.9 h | 213 GB |
| 100,000 | 2048² | 48.8 | $3.42 \times 10^{-4}$ | 300 | 880,500 | 8.9 h | 213 GB |

- **Time step:** fixed at $\min(0.7\, \Delta x / U,\ 0.25\, \Delta x^2 / \nu)$, shortened slightly to land on the output
  times. The viscous limit sets it at Re = 100 and 1000. An adaptive step at the same CFL number wouldn't gain much,
  since the lid keeps the top speed close to $U$: in the final states at Re = 5000 and 22,000 it would be less than 2%
  longer.
- **End:** the runs stop at $t = 300\, L/U$, or once $u$ has changed by less than $10^{-6}\, U$ everywhere over the last
  $10\, L/U$. Only Re = 100 and 1000 stop early. The run at Re = 5000 uses its own stopping interval of $50\, L/U$.
- **Output:** every $0.025\, L/U$ until $t = 50\, L/U$ and every $0.1\, L/U$ after that, in the same fields as above.

Run one with `RE=22000 julia --startup-file=no --project simulate_comparison.jl`. It takes its grid from the table of
runs in `lid_driven_cavity.jl` and saves in `output/Re22000`.

### Animation

`animate_comparison.jl` draws 1734 frames at 30 frames per second: a 57.8-second video at 3840 × 2160, 95 MB in AV1.
Setting `TIMES`, such as `TIMES="10,200,300"`, renders stills at those times instead.

- **Time:** the video runs at $1.5\, L/U$ per second until $t = 20\, L/U$ and speeds up to $7.5\, L/U$ per second by $t
  = 50\, L/U$, where it stays, since the vortices in the chaotic panels would jump from frame to frame any faster. It
  ends by holding the frame at $t = 300\, L/U$ for 3 seconds. On Wikipedia, `thumbtime=0:56` shows that frame until the
  reader presses play.
- **Stopped runs:** the panels at Re = 100 and 1000 stay on their last output once their runs have stopped.
- **Pixels:** each panel is 930 pixels wide. The runs on 2048² cells are averaged down to 1024² first.
- **Colors and streamlines:** the same as the lead animation, with an extra streamline at $\psi / UL = 2 \times 10^{-7}$
  at Re = 100, whose corner eddies are too weak for the others.

### Accuracy

Two different things are true of these panels. The steady flows are accurate, and the flows above Re = 8000 are still
changing when the video ends.

The steady flows at Re = 100 and 1000 agree with published values, as `check_steady.jl` shows, and their profiles are
compared under [Velocity and vorticity profiles](#velocity-and-vorticity-profiles):

| Re | Cells | $\psi$ at the primary vortex ($UL$) | Published |
|---|---|---|---|
| 100 | 512² | −0.103517 | −0.103423, Ghia et al. (1982) |
| 1000 | 512² | −0.118880 | −0.118937, Botella and Peyret (1998); −0.118888, Erturk (2009) |

At Re = 10,000, 22,000 and 100,000 the flow is still spinning up at $t = 300\, L/U$, so these panels don't show the
final state. `check_comparison.jl` prints the kinetic energy, here as its mean over successive windows, in units of
$U^2$:

| $t$ ($L/U$) | Re = 10,000 | Re = 22,000 | Re = 100,000 |
|---|---|---|---|
| 50 to 100 | 0.02888 | 0.02278 | 0.01142 |
| 100 to 150 | 0.03282 | 0.02633 | 0.01430 |
| 150 to 200 | 0.03603 | 0.02808 | 0.01612 |
| 200 to 250 | 0.03865 | 0.02966 | 0.01903 |
| 250 to 300 | 0.04056 | 0.03058 | 0.02165 |

At Re = 10,000 the oscillation already has about the published frequencies. Over $t = 200$ to $300\, L/U$ the
enstrophy's main frequency is 0.60 $U/L$, with smaller peaks at 0.16 and 0.44, against 0.61, 0.175 and 0.4375 from
Bruneau and Saad (2006).

Treat the panels at Re = 22,000 and 100,000 as illustrations. There is no finer run to compare them with, the positions
of the vortices in a chaotic flow can't converge anyway, and at Re = 100,000 the cell Reynolds number is 49, so the
scheme's numerical dissipation may matter at the smallest scales. Their character does match the literature: at $t = 10$
and $200\, L/U$ the Re = 100,000 panel looks like Bruneau's snapshots on 1024² points in Kuhlmann and Romanò (2019),
Fig. 40, mirrored because that lid moves the other way. This is a comparison by eye, not a measurement.

## Velocity and vorticity profiles

`profiles.jl` draws $u$, $v$ and $\omega$ in the steady flows at Re = 100 and 1000, on 512² cells, and Re = 5000, on
1024² cells, all with fifth-order WENO advection. Translucent lines on the $u$ and $v$ heatmaps mark the centerlines,
and the panels beside them show $u$ and $v$ along those lines and the vorticity along the lid. The vorticity colors and
axis follow $\mathrm{asinh}(\omega L / 2U)$, and the colors saturate at $\pm 30\, U/L$, as in the animations.

The figure is a PNG of 3560 × 1840 pixels and about 1 MB, because the heatmaps are raster images, and
[Help:SVG](https://commons.wikimedia.org/wiki/Help:SVG) discourages embedding those in an SVG. `profiles.jl` runs
`oxipng` on it if that's installed, which makes it about a third smaller without changing a pixel.

The markers are these published tables:

| Paper | Tables | Values | Re |
|---|---|---|---|
| Ghia et al. (1982) | I, II | $u$ and $v$ along the centerlines | 100, 1000, 5000 |
| Ghia et al. (1982) | IV | $\omega$ on the lid | 100, 1000, 5000 |
| Botella and Peyret (1998) | 9, 10 | $u$ and $v$ along the centerlines | 1000 |
| Botella and Peyret (1998) | 19 | $\omega$ on the lid, filtered with $\sigma_{12}$ | 1000 |
| Botella and Peyret (1998) | 5 | $\omega$ at the middle of the lid | 100 |
| Erturk et al. (2005) | VI, VII | $u$ and $v$ along the centerlines | 1000, 5000 |

The script also compares the vorticity along the centerlines with Botella and Peyret's Tables 9 and 10 at Re = 1000, and
at the center with their Table 3 at Re = 100, without drawing it.

- **Signs:** Botella and Peyret's lid moves in $-x$, so their values are mirrored, with $x \to L - x$, $u \to -u$ and
  $\omega \to -\omega$, and $y$ and $v$ unchanged. Ghia et al. print the vorticity on the lid as positive, the opposite
  of $\omega = \partial_x v - \partial_y u$ here, so it is negated.
- **Misprints:** the two in Ghia et al.'s Table I are at Re = 3200 and 10,000, which the figure doesn't use. Bruneau and
  Saad (2006) quote 2.26722 where Botella and Peyret's Table 9 prints 2.26772 at $y = 0.2813$, possibly a misprint in
  one of them. The comparison uses Table 9 as printed.
- **Left out:** Bruneau and Saad (2006) and Auteri et al. (2002) also tabulate the vorticity at Re = 1000, at the same
  points as Botella and Peyret. None of these papers gives it along the centerlines at Re = 5000, so the lid is the only
  vorticity comparison there.

On the walls, the vorticity is the difference between the wall velocity and the nearest cell center, half a cell away,
which is the wall stress the model itself uses. A second-order formula through the two nearest cell centers did worse in
a one-off check: along the lid at Re = 1000 it was up to 4.2% from Botella and Peyret, against 0.59%, and at $x =
0.0625\, L$ at Re = 5000 it changed by 4.2% between 512² and 1024² cells, against 2.0%.

The script prints how far the profiles are from each table. The largest differences in velocity, in units of $U$, are:

| Re | Cells | Ghia et al., $u$ and $v$ | Botella and Peyret, $u$ and $v$ | Erturk et al., $u$ and $v$ |
|---|---|---|---|---|
| 100 | 512² | 0.0050, 0.0092 | | |
| 1000 | 512² | 0.0062, 0.0180 | 0.0003, 0.0005 | 0.0005, 0.0005 |
| 5000 | 1024² | 0.0178, 0.0222 | | 0.0042, 0.0049 |

Ghia et al.'s points sit visibly off the curves near the walls at Re = 5000, where these profiles are much closer to
Erturk et al.'s tables, so that gap comes from Ghia et al.'s values and not from this run. At Re = 1000 the extrema are
$u = -0.38836\, U$ at $y = 0.1718\, L$, $v = -0.52678\, U$ at $x = 0.9092\, L$ and $v = 0.37674\, U$ at $x = 0.1579\,
L$, against Botella and Peyret's mirrored $-0.38857$ at 0.1717, $-0.52708$ at 0.9092 and $0.37694$ at 0.1578.

The largest differences in vorticity are:

| Re | Cells | On the lid, Ghia et al. | On the lid, Botella and Peyret | Along the centerlines, Botella and Peyret |
|---|---|---|---|---|
| 100 | 512² | 0.33% | 0.01%, at $x = L/2$ | $-1.1743$ against $-1.1744\, U/L$, at the center |
| 1000 | 512² | 3.6% | 0.59% | $0.014\, U/L$ vertical, $0.012\, U/L$ horizontal |
| 5000 | 1024² | 4.1% | | |

At Re = 1000, both of the largest differences along the centerlines are on the walls, at $y = L$ and $x = L$, and their
rms is $0.004\, U/L$. Ghia et al.'s value on the lid at $x = 0.0625\, L$ at Re = 1000 is itself 4.1% from Botella and
Peyret's, so a gap of that size from their table doesn't say much. At Re = 5000, the 512² and 1024² runs differ on the
lid by 2.0% at $x = 0.0625\, L$, 1.9% at $x = 0.9375\, L$ and less than 0.8% at the five points checked between them.

## Corner flows

`scraping_flow.jl` and `moffatt_eddies.jl` each draw one of the two classical flows near the corners of the cavity from
its closed-form solution, so nothing here is simulated. Both are Stokes flows, which hold close to a corner, where
viscous forces dominate. Neither figure mentions the cavity, so they can also go in articles on the flows themselves.
The colors match the schematic: blue streamlines turn clockwise and red ones counterclockwise. Each figure is an SVG of
11 or 37 kB, with real text in DejaVu Sans and no embedded raster images.

### Taylor scraping flow

This is the flow near a corner where one wall slides into another, as at the top right corner of the cavity, where the
lid slides into the side wall. In polar coordinates $(r, \phi)$ centered on the corner, with $\phi$ measured from the
sliding wall, Batchelor (1967), section 4.8, eq. (4.8.27) on p. 226, gives

```math
\psi = U r f(\phi), \qquad
f(\phi) = \frac{-\tfrac{\pi^2}{4} \sin\phi + \tfrac{\pi}{2} \phi \sin\phi + \phi \cos\phi}{\tfrac{\pi^2}{4} - 1}.
```

Since $\psi$ is proportional to $r$, the streamlines are scaled copies of one another, and the velocity depends only on
$\phi$, so it has no single value at the corner. The arrows on each dashed line show this, with their lengths
proportional to the speed.

### Moffatt eddies

This is the leading term of the flow near a corner between two stationary walls, as at the bottom corners of the cavity,
from Moffatt (1964), eq. (3.9):

```math
\psi \propto \mathrm{Re}\left\{ r^\lambda \left[ \cos\lambda\theta \cos(\lambda - 2)\alpha
    - \cos(\lambda - 2)\theta \cos\lambda\alpha \right] \right\},
```

where $\theta$ is measured from the bisector of the corner, $\alpha = \pi/4$, and $\lambda$ is a root of his eq. (3.5),
$\sin 2(\lambda - 1)\alpha = -(\lambda - 1) \sin 2\alpha$. Newton's method from the values in his Table 1 gives $\lambda
= 3.7396 + 1.1190i$. Each eddy is then $e^{\pi / \mathrm{Im}\lambda} = 16.57$ times smaller than the last, with
velocities $e^{\pi (\mathrm{Re}\lambda - 1) / \mathrm{Im}\lambda} = 2189$ times weaker. Moffatt's Table 1 prints a
slightly different root, which gives 16.3 and 2060, and Kuhlmann and Romanò (2019), p. 240, round to about 16 and 2000,
so the figure's labels say about 16, 270 and 2000. These numbers hold only for a right angle.

The three views are the same drawing, scaled by exactly 16.57 with the colors swapped, because $\psi(r/\rho, \theta) =
-\rho^{-\mathrm{Re}\lambda} \psi(r, \theta)$. Each view has its streamlines at 0.06, 0.3 and 0.65 of the peak of the
largest eddy, its edge where $\psi = 0$, and the fluid outside it at 1, 4, 10, 22 and 45 times that peak. The levels are
chosen to show the shapes, so the spacing of the lines doesn't compare speeds between the views.

## Schematic

`schematic.jl` draws the geometry, the boundary conditions and the singular corners around streamlines of a computed
steady flow. Blue streamlines turn clockwise and red ones counterclockwise, and the eddies are labelled in the notation
of Ghia, Ghia and Shin (1982). The third eddy in the bottom right corner, BR₃, is too small to draw.

The flow is at Re = 7500, which is below the Hopf bifurcation near Re = 8000 and high enough to show the other eddies.
It uses the same model as the other runs but with second-order centered advection, on 512² cells with
$\Delta t = 0.9\, \Delta x / U$, since only the final state matters. The run stopped at $t = 1450\, L/U$, when $u$ had
changed by less than $10^{-6}\, U$ over the last $50\, L/U$, after 824,905 iterations and about 20 minutes on an
RTX 4090.

| Primary vortex | $\psi$ ($UL$) | $\omega$ ($U/L$) | Center |
|---|---|---|---|
| This run, 512² cells | −0.121542 | −1.91440 | (0.5137, 0.5312) |
| Erturk et al. (2005), 513² points | −0.120395 | −1.896895 | |
| Erturk et al. (2005), extrapolated | −0.122386 | −1.926969 | |
| Ghia et al. (1982), 257² points | −0.119976 | −1.87987 | (0.5117, 0.5322) |

| Eddy | Along the bottom wall ($L$) | Ghia et al. | Along the side wall ($L$) | Ghia et al. |
|---|---|---|---|---|
| BL₁ | 0.3447 | 0.3339 | 0.2847 | 0.2793 |
| BL₂ | 0.0261 | 0.0234 | 0.0295 | 0.0254 |
| BR₁ | 0.3787 | 0.3779 | 0.4449 | 0.4375 |
| BR₂ | 0.1249 | 0.1270 | 0.0983 | 0.0938 |
| TL₁ | | | 0.3117 | 0.2993 |

The text in the SVG is real text in DejaVu Sans, which Wikimedia Commons has. librsvg, which Commons uses, and Inkscape
render the file almost identically.
