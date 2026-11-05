# Shared code for the Taylor–Green vortex in two and three dimensions: the simulation and the vorticity.

using Oceananigans
using CUDA
using Printf

output_directory = mkpath(get(ENV, "OUTPUT_DIRECTORY", joinpath(@__DIR__, "output")))
output_file(name) = joinpath(output_directory, name)

"""
    taylor_green_simulation(;
            Re = 1600,
            N = 512,
            dimensions = 3,
            stop_time = 20,
            V₀ = 1,
            L = 1
    )

A simulation of the Taylor–Green vortex at Reynolds number `Re = V₀L/ν` on `N` cells in each direction of the periodic
domain -πL ≤ x, y, z < πL, with fifth-order WENO advection and third-order Runge–Kutta time stepping, in double
precision and on the GPU if there is one. With `dimensions = 2`, it simulates the plane z = 0 on its own: the
two-dimensional Taylor–Green vortex, an exact solution.
"""
function taylor_green_simulation(;
        Re = 1600,
        N = 512,
        dimensions = 3,
        stop_time = 20,
        V₀ = 1,
        L = 1
)
    architecture = CUDA.functional() ? GPU() : CPU()
    ν = V₀ * L / Re
    extent = (-π * L, π * L)

    grid = if dimensions == 3
        RectilinearGrid(architecture;
                size = (N, N, N),
                x = extent,
                y = extent,
                z = extent,
                topology = (Periodic, Periodic, Periodic)
        )
    else
        RectilinearGrid(architecture;
                size = (N, N),
                x = extent,
                y = extent,
                topology = (Periodic, Periodic, Flat)
        )
    end

    model = NonhydrostaticModel(grid;
            advection = WENO(order = 5),
            timestepper = :RungeKutta3,
            closure = ScalarDiffusivity(; ν)
    )

    uᵢ(x, y, z = 0) = V₀ * sin(x / L) * cos(y / L) * cos(z / L)
    vᵢ(x, y, z = 0) = -V₀ * cos(x / L) * sin(y / L) * cos(z / L)
    set!(model, u = uᵢ, v = vᵢ)

    Δx = 2π * L / N
    Δt = 0.5 * Δx / V₀
    simulation = Simulation(model; Δt, stop_time, verbose = false)
    conjure_time_step_wizard!(simulation, IterationInterval(10), cfl = 0.6, max_change = 1.1)

    return simulation
end

"""
    vorticity(u, v, w)

The components of the vorticity ∇ × u of the velocity field (`u`, `v`, `w`), as operations.
"""
vorticity(u, v, w) = (∂y(w) - ∂z(v), ∂z(u) - ∂x(w), ∂x(v) - ∂y(u))
