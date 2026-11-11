# Prints how the runs behind animate_comparison.jl end, from the mean squares they saved: the kinetic energy, which is
# still rising at t = 300 L/U at the three highest Reynolds numbers, and the main frequencies in the enstrophy at
# Re = 10,000, against those of Bruneau and Saad (2006).
#
#     julia --project check_comparison.jl

include("lid_driven_cavity.jl")

using Statistics

"""
    without_drift(t, y; degree = 4)

`y` less its best fit by a polynomial in `t`, since the slow rise would otherwise swamp the oscillations.
"""
function without_drift(t, y; degree = 4)
    x = (t .- mean(t)) ./ (t[end] - t[1])
    polynomial = hcat([x .^ d for d in 0:degree]...)

    return y .- polynomial * (polynomial \ y)
end

"""
    largest_peaks(t, y; lowest = 0.1, highest = 3, count = 5)

The frequencies of the `count` largest peaks in the Hann-windowed spectrum of `y`, with their power relative to the
largest.
"""
function largest_peaks(t, y; lowest = 0.1, highest = 3, count = 5)
    wave = without_drift(t, y) .* (0.5 .- 0.5 .* cos.(2π .* (0:length(y)-1) ./ (length(y) - 1)))
    frequencies = range(lowest, highest, step = 1 / (10 * (t[end] - t[1])))
    power = [abs2(sum(wave .* cis.(-2π .* f .* t))) for f in frequencies]

    peaks = [i for i in 2:length(power)-1 if power[i] > power[i-1] && power[i] > power[i+1]]
    sort!(peaks, by = i -> -power[i])

    return [(frequencies[i], power[i] / power[first(peaks)]) for i in peaks[1:min(count, end)]]
end

for (; Re) in comparison_runs
    averages = mean_squares(run_directory(Re))
    t = averages.t
    energy = (averages.u² .+ averages.v²) ./ 2

    @printf("Re = %6d: ends at t = %6.1f with kinetic energy %.5f U²\n", Re, t[end], energy[end])

    if Re ≥ 10_000
        for from in 50:50:250
            window = from .≤ t .≤ from + 50
            @printf("    mean energy over t = %3d to %3d: %.5f U²\n", from, from + 50, mean(energy[window]))
        end
    end

    if Re == 10_000
        for from in (200, 250)
            window = from .≤ t .≤ 300
            peaks = largest_peaks(t[window], averages.ω²[window])
            peaks = join([@sprintf("%.3f (%.3f)", f, p) for (f, p) in peaks], ", ")
            println("    enstrophy over t = $from to 300, frequencies (relative power): $peaks")
        end

        println("    Bruneau and Saad (2006) give a main frequency of 0.61, and lower ones of 0.175 and 0.4375.")
    end
end
