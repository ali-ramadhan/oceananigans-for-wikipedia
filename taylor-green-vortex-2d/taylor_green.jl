# Shared code for the two-dimensional Taylor–Green vortex: the simulation, the derived fields and the plots.

using Oceananigans
using Oceananigans.Architectures: on_architecture
using CUDA
using CairoMakie
using JLD2
using Printf
using Random
using Statistics

include("video_encoding.jl")

set_theme!(fontsize = 22, Label = (; fontsize = 28, tellwidth = false))

output_directory = mkpath(joinpath(@__DIR__, "output"))
output_file(name) = joinpath(output_directory, name)

#####
##### Simulation
#####

"""
    taylor_green_simulation(;
            Re = 100,
            N = 256,
            stop_time = 100,
            U₀ = 1,
            k = 1,
            perturbation = 0,
            seed = 1,
            particles_per_side = 24
    )

A simulation of the Taylor–Green vortex at Reynolds number `Re = U₀ / (kν)` on an `N × N` grid covering one period,
0 ≤ x, y < 2π/k, with fifth-order WENO advection and third-order Runge–Kutta time stepping. It carries
`particles_per_side²` fluid particles that start on a uniform lattice. A nonzero `perturbation` adds a random velocity
whose root-mean-square speed is `perturbation` times that of the vortex.
"""
function taylor_green_simulation(;
        Re = 100,
        N = 256,
        stop_time = 100,
        U₀ = 1,
        k = 1,
        perturbation = 0,
        seed = 1,
        particles_per_side = 24
)
    architecture = CUDA.functional() ? GPU() : CPU()
    L = 2π / k
    ν = U₀ / (k * Re)

    grid = RectilinearGrid(architecture;
            size = (N, N),
            x = (0, L),
            y = (0, L),
            topology = (Periodic, Periodic, Flat)
    )

    particles = particle_lattice(architecture, L, particles_per_side)

    model = NonhydrostaticModel(grid;
            advection = WENO(order = 5),
            timestepper = :RungeKutta3,
            closure = ScalarDiffusivity(; ν),
            particles
    )

    δu, δv = random_velocity(perturbation; U₀, k, seed)
    uᵢ(x, y) = U₀ * sin(k * x) * cos(k * y) + δu(x, y)
    vᵢ(x, y) = -U₀ * cos(k * x) * sin(k * y) + δv(x, y)
    set!(model, u = uᵢ, v = vᵢ)

    Δx = L / N
    Δt = min(0.2 * Δx / U₀, 0.1 * Δx^2 / ν)
    simulation = Simulation(model; Δt, stop_time)

    # RK3 is stable for diffusion only when 8νΔt/Δx² ≲ 2.5 in 2D, so the diffusive CFL must stay below ≈ 0.3.
    conjure_time_step_wizard!(simulation, IterationInterval(10);
            cfl = 0.5,
            diffusive_cfl = 0.2,
            max_change = 1.1
    )

    return simulation
end

function particle_lattice(architecture, L, n)
    spacing = L / n
    x = [(i - 1/2) * spacing for i in 1:n, j in 1:n]
    y = [(j - 1/2) * spacing for i in 1:n, j in 1:n]
    z = zeros(n, n)

    on_grid(coordinate) = on_architecture(architecture, vec(coordinate))
    return LagrangianParticles(x = on_grid(x), y = on_grid(y), z = on_grid(z))
end

"""
    random_velocity(perturbation; U₀, k, seed)

The velocity components `δu(x, y)` and `δv(x, y)` of the random streamfunction Σ aₘₙ cos(k(mx + ny) + φₘₙ), with
wavenumbers up to 4k, whose root-mean-square speed is `perturbation` times that of the vortex.
"""
function random_velocity(perturbation; U₀, k, seed)
    rng = Xoshiro(seed)
    wavevectors = [(m, n) for m in -4:4 for n in 0:4 if n > 0 || m > 0]
    amplitudes = randn(rng, length(wavevectors))
    phases = 2π .* rand(rng, length(wavevectors))

    # The vortex has mean square speed U₀²/2, and each mode a²k²(m² + n²)/2.
    speed² = sum(a^2 * k^2 * (m^2 + n^2) for ((m, n), a) in zip(wavevectors, amplitudes))
    amplitudes .*= perturbation * U₀ / sqrt(speed²)
    modes = zip(wavevectors, amplitudes, phases)

    δu(x, y) = -sum(a * k * n * sin(k * (m * x + n * y) + φ) for ((m, n), a, φ) in modes)
    δv(x, y) = sum(a * k * m * sin(k * (m * x + n * y) + φ) for ((m, n), a, φ) in modes)

    return δu, δv
end

#####
##### Derived fields
#####

vorticity(u, v) = Field(∂x(v) - ∂y(u))

"""
    streamfunction(u, v)

The streamfunction ψ on the cell corners of a doubly periodic grid, from the volume fluxes across cell faces: v = -∂ψ/∂x
along the bottom edge, then u = ∂ψ/∂y up each column. It is exact for a discretely divergence-free velocity, and its
mean is zero.
"""
function streamfunction(u, v)
    grid = u.grid
    Δx = grid.Lx / grid.Nx
    Δy = grid.Ly / grid.Ny
    U = interior(u, :, :, 1)
    V = interior(v, :, :, 1)

    along_bottom = cumsum([0; -V[1:end-1, 1] .* Δx])
    up_columns = cumsum([zeros(grid.Nx) U[:, 1:end-1] .* Δy], dims = 2)
    Ψ = along_bottom .+ up_columns

    ψ = Field{Face, Face, Center}(grid)
    interior(ψ) .= Ψ .- mean(Ψ)
    return ψ
end

# Heatmaps of cell-centered fields cover the whole domain.
cell_centered(field) = set!(CenterField(field.grid), field)

#####
##### Fluid particles
#####

"""
    particle_trajectories(filename)

The particle positions that a `JLD2Writer` saved to `filename`, as a matrix of points with one row per particle and one
column per output.
"""
function particle_trajectories(filename)
    return jldopen(filename) do file
        iterations = sort(parse.(Int, keys(file["timeseries/t"])))
        stack(iterations) do iteration
            particles = file["timeseries/particles/$iteration"]
            Point2f.(particles.x, particles.y)
        end
    end
end

"""
    particle_trails(trajectories, frame, L; trail_length = 10)

The last `trail_length` steps of every particle up to `frame`, as a single line for `lines!`. NaNs separate the
particles and break the line where a particle crosses a periodic boundary of the domain of side `L`.
"""
function particle_trails(trajectories, frame, L; trail_length = 10)
    trails = Point2f[]
    for trajectory in eachrow(trajectories)
        for position in trajectory[max(1, frame - trail_length):frame]
            crossed_boundary = !isempty(trails) && any(abs.(position - trails[end]) .> L / 2)
            crossed_boundary && push!(trails, Point2f(NaN))
            push!(trails, position)
        end
        push!(trails, Point2f(NaN))
    end
    return trails
end

#####
##### Plots
#####

function domain_axis(position, grid;
        size,
        ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]),
        attributes...
)
    return Axis(position;
            xlabel = "x",
            ylabel = "y",
            width = size,
            height = size,
            limits = ((0, grid.Lx), (0, grid.Ly)),
            xticks = ticks,
            yticks = ticks,
            attributes...
    )
end

"""
    vorticity_panel!(ax, u, v, trajectories, frame; U₀ = 1, k = 1)

Plot output `frame[]` of the velocity time series `u` and `v` on the axis `ax`: the vorticity, the streamlines, and the
fluid particles with their trails. Lines and markers scale with the width of `ax`. Returns the heatmap, for a colorbar.
"""
function vorticity_panel!(ax, u, v, trajectories, frame; U₀ = 1, k = 1)
    ω₀ = 2k * U₀
    scale = ax.width[] / 840

    ω = @lift cell_centered(vorticity(u[$frame], v[$frame]))
    vorticity_heatmap = heatmap!(ax, ω; colormap = :balance, colorrange = (-ω₀, ω₀))

    ψ = @lift streamfunction(u[$frame], v[$frame])
    levels = @lift maximum(abs, $ψ) .* [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    visible = @lift maximum(abs, $ω) > 0.02ω₀

    contour!(ax, ψ;
            levels,
            visible,
            color = (:black, 0.35),
            linewidth = 1.5scale
    )

    trails = @lift particle_trails(trajectories, $frame, u.grid.Lx)
    lines!(ax, trails; color = (:black, 0.45), linewidth = 2scale)

    positions = @lift trajectories[:, $frame]
    scatter!(ax, positions; color = :black, markersize = 7scale)

    return vorticity_heatmap
end

"""
    save_animation(figure, frame, times, name; still_time)

Step the observable `frame` through `times` to write the lossless master `<name>_master.mkv` and the video `<name>.webm`
to the output directory, along with `<name>.png`, a still at `still_time`.
"""
function save_animation(figure, frame, times, name; still_time)
    master = record_lossless(figure, output_file("$(name)_master.mkv"), eachindex(times)) do n
        frame[] = n
    end
    encode_webm(master, output_file("$name.webm"))

    frame[] = findfirst(≥(still_time), times)
    save(output_file("$name.png"), figure; px_per_unit = 2)

    return nothing
end
