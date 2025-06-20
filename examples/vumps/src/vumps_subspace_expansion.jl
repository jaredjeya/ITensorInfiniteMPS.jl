using ITensors, ITensorMPS
using ITensorInfiniteMPS

# Alternate steps of running TDVP and increasing the bond dimension.
# Attempt increasing the bond dimension `outer_iters` number of times.
function tdvp_subspace_expansion(
  H, ψ; time_step, outer_iters, subspace_expansion_kwargs, vumps_kwargs
)
  @time for outer_iter in 1:outer_iters
    println(
      "\nIncrease bond dimension $(outer_iter) out of $(outer_iters), starting from dimension $(maxlinkdim(ψ))",
    )
    println(
      "cutoff = $(subspace_expansion_kwargs[:cutoff]), maxdim = $(subspace_expansion_kwargs[:maxdim])",
    )
    ψ = @time subspace_expansion(ψ, H; subspace_expansion_kwargs...)
    println("\nRun VUMPS with new bond dimension $(maxlinkdim(ψ))")
    ψ, _ = @time tdvp(H, ψ; time_step=time_step, vumps_kwargs...)
  end
  return ψ
end

# Alternate steps of running VUMPS and increasing the bond dimension
function vumps_subspace_expansion(
  H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs
)
  return tdvp_subspace_expansion(
    H, ψ; time_step=-Inf, outer_iters, subspace_expansion_kwargs, vumps_kwargs
  )
end
