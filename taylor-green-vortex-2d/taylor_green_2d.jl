# The two-dimensional Taylor–Green vortex, an exact solution of the Navier–Stokes equations:
#
#     u = U₀ sin(kx) cos(ky) F(t),   v = -U₀ cos(kx) sin(ky) F(t),
#     p/ρ = (U₀²/4) [cos(2kx) + cos(2ky)] F(t)²,   F(t) = exp(-2νk²t).
#
# The script simulates the flow with Oceananigans, advects Lagrangian particles, checks the numerical
# solution against the exact one, and animates the vorticity, streamlines, and fluid particles.
#
#     julia --project taylor_green_2d.jl

using Oceananigans
using Oceananigans.Architectures: on_architecture
using CUDA
using CairoMakie
using JLD2
using Printf

include("video_encoding.jl")

arch = CUDA.functional() ? GPU() : CPU()

N = 256
U₀ = 1
k = 1
ν = 0.01
Re = U₀ / (k * ν)
stop_time = 100
frame_interval = 0.25

output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)
fields_filename = joinpath(output_directory, "taylor_green_2d.jld2")
particles_filename = joinpath(output_directory, "taylor_green_2d_particles.jld2")

F(t) = exp(-2ν * k^2 * t)
exact_vorticity(x, y, t) = 2k * U₀ * sin(k * x) * sin(k * y) * F(t)
exact_pressure(x, y, t) = U₀^2 / 4 * (cos(2k * x) + cos(2k * y)) * F(t)^2
exact_kinetic_energy(t) = U₀^2 / 4 * F(t)^2

grid = RectilinearGrid(arch; size=(N, N), x=(0, 2π / k), y=(0, 2π / k), topology=(Periodic, Periodic, Flat))

particles_per_side = 24
particle_spacing = 2π / k / particles_per_side
x₀ = [(i - 1/2) * particle_spacing for i in 1:particles_per_side, j in 1:particles_per_side]
y₀ = [(j - 1/2) * particle_spacing for i in 1:particles_per_side, j in 1:particles_per_side]
particles = LagrangianParticles(x = on_architecture(arch, vec(x₀)),
                                y = on_architecture(arch, vec(y₀)),
                                z = on_architecture(arch, zeros(length(x₀))))

model = NonhydrostaticModel(grid; closure=ScalarDiffusivity(ν=ν), particles)

set!(model, u = (x, y) -> U₀ * sin(k * x) * cos(k * y),
            v = (x, y) -> -U₀ * cos(k * x) * sin(k * y))

Δx = 2π / k / N
simulation = Simulation(model; Δt = 0.2 * Δx / U₀, stop_time)
# RK3 is stable for diffusion only when 8νΔt/Δx² ≲ 2.5 in 2D, so the diffusive CFL must stay below ≈ 0.3.
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl=0.5, diffusive_cfl=0.2, max_change=1.1)

u, v, w = model.velocities
ω = ∂x(v) - ∂y(u)
p = model.pressures.pNHS

simulation.output_writers[:fields] = JLD2Writer(model, (; ω, p);
                                                filename = fields_filename,
                                                schedule = TimeInterval(frame_interval),
                                                overwrite_files = true)

particle_times = Float64[]
particle_x = Vector{Float64}[]
particle_y = Vector{Float64}[]
energy_times = Float64[]
kinetic_energies = Float64[]

function record_particles(sim)
    if isempty(particle_times) || time(sim) > particle_times[end]
        push!(particle_times, time(sim))
        push!(particle_x, Array(sim.model.particles.properties.x))
        push!(particle_y, Array(sim.model.particles.properties.y))
    end
    return nothing
end

function record_kinetic_energy(sim)
    if isempty(energy_times) || time(sim) > energy_times[end]
        u, v, w = sim.model.velocities
        push!(energy_times, time(sim))
        push!(kinetic_energies, (sum(abs2, interior(u)) + sum(abs2, interior(v))) / (2N^2))
    end
    return nothing
end

add_callback!(simulation, record_particles, TimeInterval(frame_interval))
add_callback!(simulation, record_kinetic_energy, TimeInterval(frame_interval))

record_particles(simulation)
record_kinetic_energy(simulation)
run!(simulation)

jldsave(particles_filename; particle_times, particle_x, particle_y, energy_times, kinetic_energies)

#####
##### Check the simulation against the exact solution
#####

ωt = FieldTimeSeries(fields_filename, "ω")
pt = FieldTimeSeries(fields_filename, "p")
times = ωt.times

xω, yω, _ = nodes(ωt)
xp, yp, _ = nodes(pt)
n = length(times)
ω_error = maximum(abs, interior(ωt[n], :, :, 1) .- exact_vorticity.(xω, yω', times[n])) / (2k * U₀ * F(times[n]))
p_numerical = interior(pt[n], :, :, 1)
p_exact = exact_pressure.(xp, yp', times[n])
p_error = maximum(abs, (p_numerical .- sum(p_numerical) / N^2) .- p_exact) / (U₀^2 / 2 * F(times[n])^2)
E_error = maximum(abs, kinetic_energies ./ exact_kinetic_energy.(energy_times) .- 1)
@info @sprintf("Relative errors at t = %d: vorticity %.1e, pressure %.1e; maximum kinetic energy error %.1e",
               times[n], ω_error, p_error, E_error)

#####
##### Animation
#####

set_theme!(Theme(fontsize = 22))
fig = Figure(size = (1080, 1080))

frame = Observable(1)
ωₙ = @lift ωt[$frame]
title = @lift @sprintf("2D Taylor–Green vortex, Re = %d:   t = %5.1f", Re, times[$frame])
Label(fig[0, 1:2], title, fontsize = 28, tellwidth = false)

ax_ω = Axis(fig[1, 1]; title = "Vorticity, streamlines and fluid particles", xlabel = "x", ylabel = "y",
            limits = ((0, 2π / k), (0, 2π / k)),
            xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]),
            yticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
hm_ω = heatmap!(ax_ω, ωₙ; colormap = :balance, colorrange = (-2k * U₀, 2k * U₀))
Colorbar(fig[1, 2], hm_ω; label = "ω / kU₀")

# Contours of ω are streamlines (ω = 2k²ψ); scaling the levels by the decaying amplitude
# shows that the streamline pattern never changes.
streamline_levels = @lift 2k * U₀ * F(times[$frame]) .* [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
contour!(ax_ω, ωₙ; levels = streamline_levels, color = (:black, 0.35), linewidth = 1.5)

trail_length = 10
particle_points = @lift Point2f.(particle_x[$frame], particle_y[$frame])
particle_trails = @lift begin
    trail = Point2f[]
    for p in eachindex(particle_x[1])
        for m in max(1, $frame - trail_length):$frame
            push!(trail, Point2f(particle_x[m][p], particle_y[m][p]))
        end
        push!(trail, Point2f(NaN, NaN))
    end
    trail
end
lines!(ax_ω, particle_trails; color = (:black, 0.45), linewidth = 2)
scatter!(ax_ω, particle_points; color = :black, markersize = 7)
colsize!(fig.layout, 1, Fixed(840))
rowsize!(fig.layout, 1, Fixed(840))

master = record_lossless(fig, joinpath(output_directory, "taylor_green_2d_master.mkv"), 1:length(times)) do i
    frame[] = i
end
encode_webm(master, joinpath(output_directory, "taylor_green_2d.webm"))

frame[] = findfirst(t -> t ≥ 20, times)
save(joinpath(output_directory, "taylor_green_2d.png"), fig; px_per_unit = 2)
