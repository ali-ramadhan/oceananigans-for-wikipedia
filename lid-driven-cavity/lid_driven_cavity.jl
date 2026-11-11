# Shared code for the two-dimensional lid-driven cavity: the simulation and its output, the saved runs, and the fields
# derived from them. Every run keeps the lid speed U = 1 and the side of the cavity L = 1, so everything saved is in
# units of U and L, and the code that reads it works on the unit square.

using Oceananigans
using CUDA
using JLD2
using Printf

output_directory = mkpath(get(ENV, "OUTPUT_DIRECTORY", joinpath(@__DIR__, "output")))
output_file(name) = joinpath(output_directory, name)

# The six runs of the comparison, on finer grids at higher Reynolds numbers, where the layers along the walls are
# thinner. simulate_evolution.jl makes the run at Re = 5000, and simulate_comparison.jl the others.
comparison_runs = [
        (Re = 100, N = 512),
        (Re = 1000, N = 512),
        (Re = 5000, N = 1024),
        (Re = 10_000, N = 1024),
        (Re = 22_000, N = 2048),
        (Re = 100_000, N = 2048)
]

# Each run of the comparison is saved in a folder of its own, except the one at Re = 5000, which is also the lead
# animation's and is saved in the output directory itself.
run_directory(Re) = Re == 5000 ? output_directory : joinpath(output_directory, "Re$Re")

#####
##### Simulation
#####

"""
    lid_driven_cavity_simulation(;
            Re,
            N,
            stop_time,
            advection = WENO(order = 5),
            cfl = 0.7,
            U = 1,
            L = 1
    )

A simulation of the flow in a square cavity of side `L` at Reynolds number `Re = UL/ν`, on `N × N` cells. The fluid
starts at rest and the lid, the wall y = L, moves in the +x direction at speed `U` from the first time step. It uses
third-order Runge–Kutta time stepping, in double precision and on the GPU if there is one. The time step is fixed, at
the advective CFL number `cfl` or at the viscous limit if that is shorter. The lid keeps the top speed close to `U`, so
an adaptive step would gain little.
"""
function lid_driven_cavity_simulation(;
        Re,
        N,
        stop_time,
        advection = WENO(order = 5),
        cfl = 0.7,
        U = 1,
        L = 1
)
    architecture = CUDA.functional() ? GPU() : CPU()
    ν = U * L / Re

    grid = RectilinearGrid(architecture;
            size = (N, N),
            x = (0, L),
            y = (0, L),
            topology = (Bounded, Bounded, Flat)
    )

    # The velocity normal to each wall vanishes by default, so only the tangential velocities need conditions.
    u_bcs = FieldBoundaryConditions(north = ValueBoundaryCondition(U), south = ValueBoundaryCondition(0))
    v_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0), west = ValueBoundaryCondition(0))

    model = NonhydrostaticModel(grid;
            advection,
            timestepper = :RungeKutta3,
            closure = ScalarDiffusivity(; ν),
            boundary_conditions = (u = u_bcs, v = v_bcs)
    )

    # Oceananigans' version and the model's schemes, for the record in the run's log.
    @info "Oceananigans $(pkgversion(Oceananigans)), Re = $Re on $(N)² cells, " *
          "$(summary(model.advection.momentum)) advection, $(nameof(typeof(model.timestepper))) time stepping"

    Δx = L / N
    Δt = min(cfl * Δx / U, 0.25 * Δx^2 / ν)

    return Simulation(model; Δt, stop_time, verbose = false)
end

function progress(sim)
    u, v, _ = sim.model.velocities

    @info @sprintf("t = %6.1f, iteration %8d, Δt = %.2e, max |u| = %.3f, max |v| = %.3f, wall time %s",
            time(sim),
            iteration(sim),
            sim.Δt,
            maximum(abs, u),
            maximum(abs, v),
            prettytime(sim.run_wall_time)
    )

    return nothing
end

"""
    save_evolution!(simulation, output_times; directory = output_directory)

Save the vorticity, `u` and the mean squares of the velocity and vorticity at `output_times`, in `directory`.
"""
function save_evolution!(simulation, output_times; directory = output_directory)
    model = simulation.model
    u, v, _ = model.velocities
    ω = Field(∂x(v) - ∂y(u))

    mean_square(field) = Field(Average(field^2))

    function save_output!(name, outputs; writer_options...)
        simulation.output_writers[name] = JLD2Writer(model, outputs;
                schedule = SpecifiedTimes(output_times),
                filename = joinpath(directory, "evolution_$name.jld2"),
                overwrite_files = true,
                writer_options...
        )

        return nothing
    end

    save_output!(:vorticity, (; ω); compress = false)
    save_output!(:averages, (u² = mean_square(u), v² = mean_square(v), ω² = mean_square(ω)))

    # The stream function comes from the velocity, and its smallest streamlines need double precision.
    save_output!(:velocity, (; u); compress = false, array_type = Array{Float64})

    return nothing
end

steady_state_file(directory, Re, N) = joinpath(directory, "steady_Re$(Re)_N$N.jld2")

"""
    stop_when_steady!(simulation; Re, check_interval, tolerance = 1e-6, directory = output_directory)

Every `check_interval`, save the velocity in `directory`, for an interrupted run to leave its latest state, and stop
the run once `u` has changed by less than `tolerance` everywhere since the last check.
"""
function stop_when_steady!(simulation; Re, check_interval, tolerance = 1e-6, directory = output_directory)
    u, v, _ = simulation.model.velocities
    N = simulation.model.grid.Nx
    previous_u = copy(interior(u))

    function check_and_save(sim)
        # Callbacks also run before the first time step, when nothing has changed yet.
        iteration(sim) == 0 && return nothing

        change = maximum(abs, interior(u) .- previous_u)
        copyto!(previous_u, interior(u))

        @info @sprintf("t = %6.1f, iteration %8d: largest change in u over the last %d L/U = %.2e, wall time %s",
                time(sim),
                iteration(sim),
                check_interval,
                change,
                prettytime(sim.run_wall_time)
        )

        # Named for the runs that end steady. The others leave the state they reached.
        jldsave(steady_state_file(directory, Re, N);
                u = Array(interior(u, :, :, 1)),
                v = Array(interior(v, :, :, 1)),
                Re,
                N,
                time = time(sim),
                last_change = change
        )

        change < tolerance && (sim.running = false)

        return nothing
    end

    add_callback!(simulation, check_and_save, TimeInterval(check_interval))

    return nothing
end

#####
##### Saved runs
#####

"""
    saved_series(directory, file, name)

The time series of the field `name` that `save_evolution!` wrote to `evolution_<file>.jld2` in `directory`, read from
disk one output at a time.
"""
function saved_series(directory, file, name)
    return FieldTimeSeries(joinpath(directory, "evolution_$file.jld2"), name; backend = OnDisk())
end

"""
    mean_squares(directory)

The output times of the run saved in `directory`, and the mean squares of `u`, `v` and the vorticity at those times.
"""
function mean_squares(directory)
    return jldopen(joinpath(directory, "evolution_averages.jld2")) do file
        iterations = sort(parse.(Int, keys(file["timeseries/t"])))
        saved(name) = [only(file["timeseries/$name/$i"]) for i in iterations]

        (t = saved("t"), u² = saved("u²"), v² = saved("v²"), ω² = saved("ω²"))
    end
end

"""
    steady_state(directory, Re, N)

The state that the run at Reynolds number `Re` on `N²` cells last saved in `directory`: `u` and `v` on their cell faces,
the time, and the largest change in `u` over the check interval before it.
"""
function steady_state(directory, Re, N)
    return jldopen(steady_state_file(directory, Re, N)) do file
        (u = file["u"], v = file["v"], stop_time = file["time"], last_change = file["last_change"])
    end
end

nearest(times, t) = argmin(abs.(times .- t))

#####
##### Derived fields
#####

"""
    stream_function(u)

The stream function ψ on the cell corners, from `u` on its cell faces: ψ vanishes on the bottom wall and ∂ψ/∂y = u.
"""
function stream_function(u)
    N = size(u, 2)
    Δ = 1 / N
    ψ = zeros(N + 1, N + 1)
    ψ[:, 2:end] .= cumsum(u, dims = 2) .* Δ

    return ψ
end

"""
    primary_vortex(ψ)

The strength of the primary vortex, which is the minimum of the stream function `ψ`, the position of the cell corner
where it is reached, and the index of that corner.
"""
function primary_vortex(ψ)
    Δ = 1 / (size(ψ, 1) - 1)
    strength, corner = findmin(ψ)
    i, j = Tuple(corner)

    return (; strength, x = (i - 1) * Δ, y = (j - 1) * Δ, corner)
end

"""
    vorticity_on_corners(u, v)

The vorticity on the cell corners, from `u` and `v` on their cell faces. The derivatives normal to the walls are the
difference between the wall value and the nearest cell center, half a cell away, which is the wall stress the model
itself uses. Along the lid at Re = 1000 this stays within 0.6% of the table of Botella and Peyret (1998), where a
second-order formula through the two nearest cell centers is off by up to 4.2%. The two corners of the lid are singular,
and those and the two bottom corners just repeat a neighbor, for heatmaps.
"""
function vorticity_on_corners(u, v)
    N = size(u, 2)
    Δ = 1 / N
    ω = zeros(N + 1, N + 1)

    for j in 2:N, i in 2:N
        ω[i, j] = (v[i, j] - v[i-1, j]) / Δ - (u[i, j] - u[i, j-1]) / Δ
    end

    # The derivative going into the fluid from a wall with the value `wall`, to the cell center `inside` it.
    into_fluid(wall, inside) = 2 * (inside - wall) / Δ

    for i in 2:N
        ω[i, 1] = -into_fluid(0, u[i, 1])
        ω[i, N+1] = into_fluid(1, u[i, N])
    end

    for j in 2:N
        ω[1, j] = into_fluid(0, v[1, j])
        ω[N+1, j] = -into_fluid(0, v[N, j])
    end

    ω[1, 1], ω[N+1, 1] = ω[2, 1], ω[N, 1]
    ω[1, N+1], ω[N+1, N+1] = ω[2, N+1], ω[N, N+1]

    return ω
end

"""
    cell_average(corners)

The average over each cell of a field that lives on the cell corners.
"""
function cell_average(corners)
    return (corners[1:end-1, 1:end-1] .+ corners[2:end, 1:end-1] .+
            corners[1:end-1, 2:end] .+ corners[2:end, 2:end]) ./ 4
end

"""
    at_most(A, n)

`A` averaged over blocks of 2 × 2 values until it has at most `n` values along each side.
"""
function at_most(A, n)
    size(A, 1) ≤ n && return A

    halved = (A[1:2:end-1, 1:2:end-1] .+ A[2:2:end, 1:2:end-1] .+
              A[1:2:end-1, 2:2:end] .+ A[2:2:end, 2:2:end]) ./ 4

    return at_most(halved, n)
end
