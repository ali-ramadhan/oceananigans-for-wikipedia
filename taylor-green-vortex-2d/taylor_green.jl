# Shared code for the two-dimensional Taylor–Green vortex: the simulation, the streamfunction, and the vorticity panel
# with streamlines and fluid particles. The exact solution is
#
#     u = U₀ sin(kx) cos(ky) F(t),   v = -U₀ cos(kx) sin(ky) F(t),
#     p/ρ = (U₀²/4) [cos(2kx) + cos(2ky)] F(t)²,   F(t) = exp(-2νk²t),
#
# with streamfunction ψ = (U₀/k) sin(kx) sin(ky) F(t) and vorticity ω = 2k²ψ.

using Oceananigans
using Oceananigans.Architectures: on_architecture
using CUDA
using CairoMakie
using JLD2
using Printf
using Random

"""
    taylor_green_simulation(; Re = 100, N = 256, stop_time = 100, U₀ = 1, k = 1, perturbation = 0, seed = 1,
                              particles_per_side = 24)

A simulation of the Taylor–Green vortex at Reynolds number `Re = U₀ / (kν)` on an `N × N` grid covering one period,
0 ≤ x, y < 2π/k, with fifth-order WENO advection and third-order Runge–Kutta time stepping. It carries
`particles_per_side²` fluid particles that start on a uniform lattice. A nonzero `perturbation` adds a random velocity
field with wavenumbers up to 4k, from the stream function Σ aₘₙ cos(k(mx + ny) + φₘₙ), whose root-mean-square speed
is `perturbation` times that of the vortex.
"""
function taylor_green_simulation(; Re = 100, N = 256, stop_time = 100, U₀ = 1, k = 1, perturbation = 0, seed = 1,
                                   particles_per_side = 24)
    architecture = CUDA.functional() ? GPU() : CPU()
    ν = U₀ / (k * Re)
    L = 2π / k
    grid = RectilinearGrid(architecture; size = (N, N), x = (0, L), y = (0, L), topology = (Periodic, Periodic, Flat))

    particle_spacing = L / particles_per_side
    x₀ = [(i - 1/2) * particle_spacing for i in 1:particles_per_side, j in 1:particles_per_side]
    y₀ = [(j - 1/2) * particle_spacing for i in 1:particles_per_side, j in 1:particles_per_side]
    particles = LagrangianParticles(x = on_architecture(architecture, vec(x₀)),
                                    y = on_architecture(architecture, vec(y₀)),
                                    z = on_architecture(architecture, zeros(particles_per_side^2)))

    model = NonhydrostaticModel(grid; advection = WENO(order = 5), timestepper = :RungeKutta3,
                                closure = ScalarDiffusivity(; ν), particles)

    rng = Xoshiro(seed)
    modes = [(m, n) for m in -4:4 for n in 0:4 if n > 0 || m > 0]
    a = randn(rng, length(modes))
    φ = 2π .* rand(rng, length(modes))
    # The vortex has mean square speed U₀²/2, and each mode a²k²(m² + n²)/2.
    a .*= perturbation * U₀ / sqrt(sum(a[j]^2 * k^2 * (m^2 + n^2) for (j, (m, n)) in enumerate(modes)))

    δu(x, y) = -sum(a[j] * k * n * sin(k * (m * x + n * y) + φ[j]) for (j, (m, n)) in enumerate(modes))
    δv(x, y) = sum(a[j] * k * m * sin(k * (m * x + n * y) + φ[j]) for (j, (m, n)) in enumerate(modes))
    uᵢ(x, y) = U₀ * sin(k * x) * cos(k * y) + δu(x, y)
    vᵢ(x, y) = -U₀ * cos(k * x) * sin(k * y) + δv(x, y)
    set!(model, u = uᵢ, v = vᵢ)

    Δx = L / N
    simulation = Simulation(model; Δt = min(0.2 * Δx / U₀, 0.1 * Δx^2 / ν), stop_time)
    # RK3 is stable for diffusion only when 8νΔt/Δx² ≲ 2.5 in 2D, so the diffusive CFL must stay below ≈ 0.3.
    conjure_time_step_wizard!(simulation, IterationInterval(10), cfl = 0.5, diffusive_cfl = 0.2, max_change = 1.1)

    return simulation
end

# A plain Field rather than a computed one, whose type would change with every snapshot of u and v.
vorticity(u, v) = set!(Field{Face, Face, Center}(u.grid), Field(∂x(v) - ∂y(u)))

"""
    streamfunction(u, v)

The streamfunction ψ on the cell corners of a doubly periodic grid, with u = ∂ψ/∂y and v = -∂ψ/∂x, from the volume
fluxes across cell faces. It is exact for a discretely divergence-free velocity, and its mean is zero.
"""
function streamfunction(u, v)
    grid = u.grid
    Δx = grid.Lx / grid.Nx
    Δy = grid.Ly / grid.Ny
    U = interior(u, :, :, 1)
    V = interior(v, :, :, 1)
    ψ₁ = cumsum([0; -V[1:end-1, 1] .* Δx])
    Ψ = ψ₁ .+ [zeros(grid.Nx) cumsum(U[:, 1:end-1] .* Δy, dims = 2)]

    ψ = Field{Face, Face, Center}(grid)
    interior(ψ, :, :, 1) .= Ψ .- sum(Ψ) / length(Ψ)
    return ψ
end

# Interpolate to cell centres, so that heatmaps cover the whole domain.
cell_centered(field) = set!(CenterField(field.grid), field)

"""
    particle_positions(filename)

The particle positions written by a `JLD2Writer` to `filename`, as `x` and `y`, each with one vector per output.
"""
function particle_positions(filename)
    return jldopen(filename) do file
        iterations = sort(parse.(Int, keys(file["timeseries/t"])))
        x = [file["timeseries/particles/$i"].x for i in iterations]
        y = [file["timeseries/particles/$i"].y for i in iterations]
        (; x, y)
    end
end

"""
    vorticity_panel!(ax, u, v, particles, n; U₀ = 1, k = 1, trail_length = 10, scale = 1)

Plot output `n[]` of the velocity time series `u` and `v` on the axis `ax`: the vorticity, the streamlines, and the
fluid `particles` (from [`particle_positions`](@ref)) with their trails. Line widths and marker sizes are multiplied
by `scale`, for axes smaller or larger than 840 pixels. Returns the heatmap, for a colorbar.
"""
function vorticity_panel!(ax, u, v, particles, n; U₀ = 1, k = 1, trail_length = 10, scale = 1)
    ω = @lift vorticity(u[$n], v[$n])
    ψ = @lift streamfunction(u[$n], v[$n])
    heatmap = heatmap!(ax, @lift(cell_centered($ω)); colormap = :balance, colorrange = (-2k * U₀, 2k * U₀))

    # Streamlines are contours of ψ at fixed fractions of its maximum, so that the pattern stays visible as the flow
    # decays. They are hidden once the vorticity has decayed to 2% of its initial maximum.
    levels = @lift maximum(abs, $ψ) .* [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    visible = @lift maximum(abs, $ω) > 0.02 * 2k * U₀
    contour!(ax, ψ; levels, visible, color = (:black, 0.35), linewidth = 1.5scale)

    L = u.grid.Lx
    trails = @lift begin
        trail = Point2f[]
        for p in eachindex(particles.x[1])
            for m in max(1, $n - trail_length):$n
                xₘ = particles.x[m][p]
                yₘ = particles.y[m][p]
                # Break the trail where the particle crosses a periodic boundary.
                if !isempty(trail) && (abs(xₘ - trail[end][1]) > L / 2 || abs(yₘ - trail[end][2]) > L / 2)
                    push!(trail, Point2f(NaN, NaN))
                end
                push!(trail, Point2f(xₘ, yₘ))
            end
            push!(trail, Point2f(NaN, NaN))
        end
        trail
    end
    lines!(ax, trails; color = (:black, 0.45), linewidth = 2scale)
    scatter!(ax, @lift(Point2f.(particles.x[$n], particles.y[$n])); color = :black, markersize = 7scale)

    return heatmap
end
