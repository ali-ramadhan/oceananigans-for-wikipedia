# Checks the simulation of the evolution on 1024² cells against the same run on 512² cells, at the same output times.
# check_steady.jl compares the final state with published values. Run these first:
#
#     julia --startup-file=no --project simulate_evolution.jl
#     OUTPUT_DIRECTORY=output/resolution_512 RESOLUTION=512 julia --startup-file=no --project simulate_evolution.jl
#
# Then:
#
#     julia --project check_evolution.jl

include("lid_driven_cavity.jl")

Re = 5000
coarse_directory = joinpath(output_directory, "resolution_512")
check_times = (1, 5, 10, 15, 20, 30, 50, 80, 120, 200, 400, 900)

root_mean_square(A) = sqrt(sum(abs2, A) / length(A))

#####
##### Kinetic energy
#####

fine_averages = mean_squares(output_directory)
coarse_averages = mean_squares(coarse_directory)

kinetic_energy(averages) = (averages.u² .+ averages.v²) ./ 2

fine_energy = kinetic_energy(fine_averages)
coarse_energy = kinetic_energy(coarse_averages)

println("Kinetic energy per unit mass, in units of U²:")

for t in check_times
    n = nearest(fine_averages.t, t)
    m = nearest(coarse_averages.t, t)

    @printf("  t = %5.1f: 1024² %.6f, 512² %.6f, difference %5.2f%%\n",
            fine_averages.t[n],
            fine_energy[n],
            coarse_energy[m],
            100 * (coarse_energy[m] / fine_energy[n] - 1)
    )
end

#####
##### The velocity and the primary vortex
#####

fine_u = saved_series(output_directory, "velocity", "u")
coarse_u = saved_series(coarse_directory, "velocity", "u")

# The faces of the coarse grid are also faces of the fine grid. Along each coarse face, average the two fine values.
coarsened(u) = (u[1:2:end, 1:2:end] .+ u[1:2:end, 2:2:end]) ./ 2

println("\nThe velocity u and the primary vortex:")

for t in check_times
    fine = interior(fine_u[nearest(fine_u.times, t)], :, :, 1)
    coarse = interior(coarse_u[nearest(coarse_u.times, t)], :, :, 1)

    difference = root_mean_square(coarsened(fine) .- coarse) / root_mean_square(coarse)

    fine_vortex = primary_vortex(stream_function(fine))
    coarse_vortex = primary_vortex(stream_function(coarse))

    @printf("  t = %5.1f: u differs by %5.2f%% of its rms;", t, 100 * difference)

    @printf(" 1024² ψ = %.5f at (%.3f, %.3f), 512² ψ = %.5f at (%.3f, %.3f), %.3f L apart\n",
            fine_vortex.strength,
            fine_vortex.x,
            fine_vortex.y,
            coarse_vortex.strength,
            coarse_vortex.x,
            coarse_vortex.y,
            hypot(fine_vortex.x - coarse_vortex.x, fine_vortex.y - coarse_vortex.y)
    )
end

#####
##### Richardson extrapolation of the final state
#####

final_strength(directory, N) = primary_vortex(stream_function(steady_state(directory, Re, N).u)).strength

ψ₁₀₂₄ = final_strength(output_directory, 1024)
ψ₅₁₂ = final_strength(coarse_directory, 512)

@printf("\nFinal primary vortex: 1024² ψ = %.6f, 512² ψ = %.6f, extrapolated at second order ψ = %.6f\n",
        ψ₁₀₂₄,
        ψ₅₁₂,
        (4ψ₁₀₂₄ - ψ₅₁₂) / 3
)
