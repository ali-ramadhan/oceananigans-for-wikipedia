# Shared set-up for the two-dimensional Taylor–Green vortex: the simulation, the frames recorded for the
# animations, and the vorticity panel with streamlines and fluid particles. The exact solution is
#
#     u = U₀ sin(kx) cos(ky) F(t),   v = -U₀ cos(kx) sin(ky) F(t),
#     p/ρ = (U₀²/4) [cos(2kx) + cos(2ky)] F(t)²,   F(t) = exp(-2νk²t),
#
# with streamfunction ψ = (U₀/k) sin(kx) sin(ky) F(t) and vorticity ω = 2k²ψ.

using Oceananigans
using Oceananigans.Architectures: on_architecture
using CUDA
using CairoMakie
using Printf
using Random

const U₀ = 1
const k = 1
const L = 2π / k

"""
    taylor_green_simulation(Re; N, stop_time, perturbation = 0, seed = 1)

A simulation of the Taylor–Green vortex at Reynolds number `Re = U₀ / (kν)` on an `N × N` grid covering one
period, 0 ≤ x, y < 2π/k, with 576 fluid particles that start on a uniform 24 × 24 lattice. A nonzero
`perturbation` adds a random velocity field with wavenumbers up to 4k, from the stream function
Σ aₘₙ cos(k(mx + ny) + φₘₙ), whose root-mean-square speed is `perturbation` times that of the vortex.
"""
function taylor_green_simulation(Re; N, stop_time, perturbation = 0, seed = 1)
    arch = CUDA.functional() ? GPU() : CPU()
    ν = U₀ / (k * Re)
    grid = RectilinearGrid(arch; size = (N, N), x = (0, L), y = (0, L), topology = (Periodic, Periodic, Flat))

    n = 24
    x₀ = [(i - 1/2) * L / n for i in 1:n, j in 1:n]
    y₀ = [(j - 1/2) * L / n for i in 1:n, j in 1:n]
    particles = LagrangianParticles(x = on_architecture(arch, vec(x₀)),
                                    y = on_architecture(arch, vec(y₀)),
                                    z = on_architecture(arch, zeros(n^2)))

    model = NonhydrostaticModel(grid; closure = ScalarDiffusivity(; ν), particles)

    rng = Xoshiro(seed)
    modes = [(m, n) for m in -4:4 for n in 0:4 if n > 0 || m > 0]
    a = randn(rng, length(modes))
    φ = 2π .* rand(rng, length(modes))
    # The vortex has mean square speed U₀²/2, and each mode a²k²(m² + n²)/2.
    a .*= perturbation * U₀ / sqrt(sum(a[j]^2 * k^2 * (m^2 + n^2) for (j, (m, n)) in enumerate(modes)))
    δu(x, y) = -sum(a[j] * k * n * sin(k * (m * x + n * y) + φ[j]) for (j, (m, n)) in enumerate(modes))
    δv(x, y) = sum(a[j] * k * m * sin(k * (m * x + n * y) + φ[j]) for (j, (m, n)) in enumerate(modes))

    xu, yu, _ = nodes(model.velocities.u)
    xv, yv, _ = nodes(model.velocities.v)
    u₀ = [U₀ * sin(k * x) * cos(k * y) + (perturbation == 0 ? 0 : δu(x, y)) for x in xu, y in yu]
    v₀ = [-U₀ * cos(k * x) * sin(k * y) + (perturbation == 0 ? 0 : δv(x, y)) for x in xv, y in yv]
    set!(model, u = reshape(u₀, N, N, 1), v = reshape(v₀, N, N, 1))

    Δx = L / N
    simulation = Simulation(model; Δt = min(0.2 * Δx / U₀, 0.1 * Δx^2 / ν), stop_time)
    # RK3 is stable for diffusion only when 8νΔt/Δx² ≲ 2.5 in 2D, so the diffusive CFL must stay below ≈ 0.3.
    conjure_time_step_wizard!(simulation, IterationInterval(10), cfl = 0.5, diffusive_cfl = 0.2, max_change = 1.1)

    return simulation
end

"""
    streamfunction(u, v, Δ)

The streamfunction at the cell corners of a doubly periodic C grid with spacing `Δ`, with u = ∂ψ/∂y and
v = -∂ψ/∂x, from the volume fluxes across cell faces. It is exact for a discretely divergence-free velocity,
and its mean is zero.
"""
function streamfunction(u, v, Δ)
    ψ = similar(u)
    ψ[1, 1] = 0
    for i in 1:size(u, 1) - 1
        ψ[i+1, 1] = ψ[i, 1] - v[i, 1] * Δ
    end
    ψ[:, 2:end] .= ψ[:, 1] .+ cumsum(u[:, 1:end-1], dims = 2) .* Δ
    return ψ .- sum(ψ) / length(ψ)
end

block_average(a, f) = f == 1 ? a : [sum(@view a[(i-1)*f+1:i*f, (j-1)*f+1:j*f]) / f^2
                                     for i in 1:size(a, 1) ÷ f, j in 1:size(a, 2) ÷ f]

"""
    record_frames!(simulation, interval; primitive = false, resolution = nothing, T = Float64)

Every `interval`, save the vorticity ω and the streamfunction ψ on the cell corners, the particle positions,
and with `primitive = true` the velocity (u, v) and the pressure p. With `resolution`, ω and ψ are averaged
over blocks of grid points to at most `resolution × resolution` points. Returns the frames as a `NamedTuple`,
which fills up as the simulation runs.
"""
function record_frames!(simulation, interval; primitive = false, resolution = nothing, T = Float64)
    model = simulation.model
    u, v, _ = model.velocities
    ω = Field(∂x(v) - ∂y(u))
    N = size(model.grid, 1)
    Δ = L / N
    f = isnothing(resolution) ? 1 : max(1, N ÷ resolution)

    frames = (; x = [((i - 1) * f + (f - 1) / 2) * Δ for i in 1:N ÷ f], t = Float64[], ω = Matrix{T}[], ψ = Matrix{T}[],
                u = Matrix{T}[], v = Matrix{T}[], p = Matrix{T}[],
                particle_x = Vector{Float64}[], particle_y = Vector{Float64}[])

    function record(sim)
        isempty(frames.t) || time(sim) > frames.t[end] || return nothing
        compute!(ω)
        uₕ = Array(interior(u, :, :, 1))
        vₕ = Array(interior(v, :, :, 1))
        push!(frames.t, time(sim))
        push!(frames.ω, T.(block_average(Array(interior(ω, :, :, 1)), f)))
        push!(frames.ψ, T.(block_average(streamfunction(uₕ, vₕ, Δ), f)))
        if primitive
            p = Array(interior(model.pressures.pNHS, :, :, 1))
            push!(frames.u, uₕ)
            push!(frames.v, vₕ)
            push!(frames.p, p .- sum(p) / length(p))
        end
        push!(frames.particle_x, Array(model.particles.properties.x))
        push!(frames.particle_y, Array(model.particles.properties.y))
        return nothing
    end

    add_callback!(simulation, record, TimeInterval(interval))
    record(simulation)
    return frames
end

# Append a periodic copy of the first row and column of a field, and one more point to its coordinates (as a
# range, which CairoMakie draws as an image), so that heatmaps and contours reach the edges of the domain.
periodic(x::AbstractVector) = range(x[1], step = x[2] - x[1], length = length(x) + 1)
periodic(a::AbstractMatrix) = a[[1:end; 1], [1:end; 1]]

"""
    vorticity_panel!(ax, frames, n; trail_length = 10, scale = 1)

Plot frame `n[]` on the axis `ax`: the vorticity, the streamlines, and the fluid particles with their trails.
Line widths and marker sizes are multiplied by `scale`, for axes smaller or larger than 840 pixels.
Returns the heatmap, for a colorbar.
"""
function vorticity_panel!(ax, frames, n; trail_length = 10, scale = 1)
    x = periodic(frames.x)
    ω = @lift periodic(frames.ω[$n])
    ψ = @lift periodic(frames.ψ[$n])
    heatmap = heatmap!(ax, x, x, ω; colormap = :balance, colorrange = (-2k * U₀, 2k * U₀))

    # Streamlines are contours of ψ at fixed fractions of its maximum, so that the pattern stays visible as the
    # flow decays. They are hidden once the vorticity has decayed to 2% of its initial maximum, when the
    # flow is too weak to see.
    levels = @lift maximum(abs, frames.ψ[$n]) .* [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    visible = @lift maximum(abs, frames.ω[$n]) > 0.02 * 2k * U₀
    contour!(ax, x, x, ψ; levels, visible, color = (:black, 0.35), linewidth = 1.5scale)

    # Trails are broken where a particle crosses a periodic boundary.
    trails = @lift begin
        trail = Point2f[]
        for p in eachindex(frames.particle_x[1])
            for m in max(1, $n - trail_length):$n
                xₘ, yₘ = frames.particle_x[m][p], frames.particle_y[m][p]
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
    scatter!(ax, @lift(Point2f.(frames.particle_x[$n], frames.particle_y[$n])); color = :black, markersize = 7scale)

    return heatmap
end
