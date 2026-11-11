# Checks a steady flow from simulate_steady.jl, simulate_evolution.jl or simulate_comparison.jl against the published
# values: the primary vortex, and the sizes of the corner eddies along the walls.
#
# RE (100, 1000, 5000 or 7500, default 7500) and RESOLUTION (default 512) pick the flow, and OUTPUT_DIRECTORY (default
# output) is the folder it was saved in:
#
#     julia --project check_steady.jl
#     RE=5000 RESOLUTION=1024 julia --project check_steady.jl
#     RE=100 OUTPUT_DIRECTORY=output/Re100 julia --project check_steady.jl

include("lid_driven_cavity.jl")
include("published_results.jl")

Re = parse(Int, get(ENV, "RE", "7500"))
N = parse(Int, get(ENV, "RESOLUTION", "512"))

published = published_vortices[Re]
(; u, v, stop_time, last_change) = steady_state(output_directory, Re, N)

Δ = 1 / N
ψ = stream_function(u)

@printf("Stopped at t = %d L/U, when u had changed by at most %.1e since the last check.\n", stop_time, last_change)
@printf("The stream function on the lid, which should vanish, is at most %.1e.\n\n", maximum(abs, ψ[:, end]))

#####
##### The primary vortex
#####

vortex = primary_vortex(ψ)
ω = vorticity_on_corners(u, v)[vortex.corner]

println("Primary vortex:")

@printf("  %-36s ψ = %.6f, ω = %.5f, at (%.4f, %.4f)\n",
        "This run, $(N)² cells",
        vortex.strength,
        ω,
        vortex.x,
        vortex.y
)

for (source, ψ_published, ω_published, center) in published.primary
    @printf("  %-36s ψ = %.6f, ω = %.5f %s\n", source, ψ_published, ω_published, isempty(center) ? "" : "at $center")
end

println()

#####
##### The corner eddies
#####

"""
    sign_changes(values, spacing)

The positions along a line, measured from its start, where `values` changes sign.
"""
function sign_changes(values, spacing)
    positions = Float64[]

    for k in 1:length(values)-1
        if values[k] * values[k+1] < 0
            push!(positions, (k - 1 + values[k] / (values[k] - values[k+1])) * spacing)
        end
    end

    return positions
end

# One cell off each wall, ψ has the sign of the circulation next to that wall. The walls themselves have ψ = 0.
along_bottom = sign_changes(ψ[2:N, 2], Δ) .+ Δ
along_left = sign_changes(ψ[2, 2:N], Δ) .+ Δ
along_right = sign_changes(ψ[N, 2:N], Δ) .+ Δ

println("Where the circulation next to each wall changes sign, measured from the bottom left corner:")
println("  bottom wall, x = ", round.(along_bottom, digits = 4))
println("  left wall,   y = ", round.(along_left, digits = 4))
println("  right wall,  y = ", round.(along_right, digits = 4))
println()
println("Sizes along the walls, with Ghia et al. (1982), Table V, in brackets:")

# The first eddy in each bottom corner ends where the primary vortex takes over, whichever smaller eddies lie beyond it.
@printf("  BL₁: H = %.4f [%.4f], V = %.4f [%.4f]\n",
        maximum(filter(<(0.5), along_bottom)),
        published.BL₁[1],
        maximum(filter(<(0.5), along_left)),
        published.BL₁[2]
)

@printf("  BR₁: H = %.4f [%.4f], V = %.4f [%.4f]\n",
        1 - minimum(filter(>(0.5), along_bottom)),
        published.BR₁[1],
        maximum(along_right),
        published.BR₁[2]
)

# The smaller eddies, at the Reynolds numbers where published_results.jl has them.
if hasproperty(published, :TL₁)
    @printf("  BL₂: H = %.4f [%.4f], V = %.4f [%.4f]\n",
            along_bottom[1],
            published.BL₂[1],
            along_left[1],
            published.BL₂[2]
    )

    @printf("  BR₂: H = %.4f [%.4f], V = %.4f [%.4f]\n",
            1 - along_bottom[4],
            published.BR₂[1],
            along_right[end-1],
            published.BR₂[2]
    )

    # They measure TL₁ between its two ends on the wall, since it stops short of the lid.
    @printf("  TL₁: V = %.4f [%.4f]\n", along_left[4] - along_left[3], published.TL₁)
end
