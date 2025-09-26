using ITensorMPS: ITensorMPS, linkinds, tdvp, update_observer!, checkdone!
using ITensors: δ, dag, noprime, prime
using ITensors.NDTensors: denseblocks
using KrylovKit: exponentiate, KrylovDefaults

struct Hᶜ{T}
  ∑h::InfiniteSum{T}
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

struct Hᴬᶜ{T}
  ∑h::InfiniteSum{T}
  Hᴸ::InfiniteMPS
  Hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (H::Hᶜ)(v)
  Hᶜv = H * v
  ## return Hᶜv * δˡ * δʳ
  return noprime(Hᶜv)
end

function (H::Hᴬᶜ)(v)
  Hᴬᶜv = H * v
  ## return Hᶜv * δˡ⁻¹ * δˢ * δʳ
  return noprime(Hᴬᶜv)
end

# Struct for use in linear system solver
struct Aᴸ
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴸ)(x)
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  dtype = ITensors.datatype(ψ)

  N = length(ψ)
  #@assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = translatecell(translator(ψ), x, -1)
  for k in (n - N + 1):n
    xT = xT * ψ.AL[k] * ψ̃.AL[k]
  end
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * dag(δʳ) * adapt(dtype, denseblocks(δˡ))
  return xT - xR
end

function left_environment(hᴸ, 𝕙ᴸ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  Aᴺ = Aᴸ(ψ, N)
  Hᴸᴺ¹, info = linsolve(Aᴺ, 𝕙ᴸ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴸ[N] = Hᴸᴺ¹
  for n in 1:(N - 1)
    Hᴸ[n] = Hᴸ[n - 1] * ψ.AL[n] * ψ̃.AL[n] + hᴸ[n]
  end
  return Hᴸ
end

# Struct for use in linear system solver
struct Aᴿ
  hᴿ::InfiniteMPS
  ψ::InfiniteCanonicalMPS
  n::Int
end

function (A::Aᴿ)(x)
  hᴿ = A.hᴿ
  ψ = A.ψ
  ψᴴ = dag(ψ)
  ψ′ = ψᴴ'
  ψ̃ = prime(linkinds, ψᴴ)
  n = A.n

  dtype = ITensors.datatype(ψ)

  N = length(ψ)
  @assert n == N

  l = linkinds(only, ψ.AL)
  l′ = linkinds(only, ψ′.AL)
  r = linkinds(only, ψ.AR)
  r′ = linkinds(only, ψ′.AR)

  xT = x
  for k in reverse(1:N)
    xT = xT * ψ.AR[k] * ψ̃.AR[k]
  end
  xT = translatecell(translator(ψ), xT, 1)
  δˡ = δ(l[n], l′[n])
  δʳ = δ(r[n], r′[n])
  xR = x * ψ.C[n] * ψ′.C[n] * δˡ * adapt(dtype, denseblocks(dag(δʳ)))
  return xT - xR
end

function right_environment(hᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  # XXX: replace with `nsites`
  #N = nsites(ψ)
  N = length(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, hᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

# TODO Generate all environments, why? Only one is needed in the sequential version
function right_environment(hᴿ, 𝕙ᴿ, ψ; tol=1e-15)
  ψ̃ = prime(linkinds, dag(ψ))
  N = nsites(ψ)

  A = Aᴿ(hᴿ, ψ, N)
  Hᴿᴺ¹, info = linsolve(A, 𝕙ᴿ[N], 1, -1; tol=tol)
  # Get the rest of the environments in the unit cell
  Hᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  Hᴿ[N] = Hᴿᴺ¹
  for n in reverse(1:(N - 1))
    Hᴿ[n] = Hᴿ[n + 1] * ψ.AR[n + 1] * ψ̃.AR[n + 1] + hᴿ[n]
  end
  return Hᴿ
end

function tdvp_iteration(args...; multisite_update_alg="sequential", kwargs...)
  if multisite_update_alg == "sequential"
    return tdvp_iteration_sequential(args...; kwargs...)
  elseif multisite_update_alg == "parallel"
    return tdvp_iteration_parallel(args...; kwargs...)
  else
    error(
      "Multisite update algorithm multisite_update_alg = $multisite_update_alg not supported, use \"parallel\" or \"sequential\"",
    )
  end
end

function ITensorMPS.tdvp(
  solver::Function,
  ∑h,
  ψ;
  eager=true,
  maxiter=10,
  tol=1e-8,
  outputlevel=1,
  multisite_update_alg="sequential",
  time_step,
  init_time=0.0,
  solver_tol=(x -> x / 100),
  (observer!)=ITensorMPS.default_observer(),
  checkdone=ITensorMPS.default_checkdone(),
  save_func=nothing,
  catch_interrupt=false,
  measure_every=1,
)
  N = nsites(ψ)
  (ϵᴸ!) = fill(tol, nsites(ψ))
  (ϵᴿ!) = fill(tol, nsites(ψ))
  ϵᵖʳᵉˢ = Inf

  if outputlevel > 0
    println("Running VUMPS with multisite_update_alg = $multisite_update_alg")
    flush(stdout)
    flush(stderr)
  end

  cur_time = init_time
  start_iter = round(Int, cur_time / time_step)

  try
    iteration_time = 0.0
    for iter in 1:maxiter
      iteration_time += @elapsed ψ, (eᴸ, eᴿ) = tdvp_iteration(
        solver,
        ∑h,
        ψ;
        (ϵᴸ!)=(ϵᴸ!),
        (ϵᴿ!)=(ϵᴿ!),
        multisite_update_alg=multisite_update_alg,
        solver_tol=solver_tol,
        time_step=time_step,
        eager,
      )
      cur_time += time_step

      if measure_every == 1 || (iter + start_iter) % measure_every == 0
        update_observer!(observer!; state=ψ, operator=∑h, sweep=iter+start_iter, outputlevel, cur_time)
      end

      ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
      maxdimψ = maxlinkdim(ψ[0:(N + 1)])
      if outputlevel > 0 && (measure_every == 1 || (iter + start_iter) % measure_every == 0)
        @printf(
          "VUMPS iteration %d (out of maximum %d). Bond dimension = %d, energy = (%s, %s), ϵᵖʳᵉˢ = %.3e, tol = %.1e, iteration time = %.2f seconds\n",
          iter + start_iter,
          maxiter + start_iter,
          maxdimψ,
          round.(real(eᴸ); digits=6),
          round.(real(eᴿ); digits=6),
          ϵᵖʳᵉˢ,
          tol,
          iteration_time
        )
        iteration_time = 0.0
        flush(stdout)
        flush(stderr)
      end

      if ϵᵖʳᵉˢ < tol
        if outputlevel > 0
          @printf "Precision error %.3e reached tolerance %.1e, stopping VUMPS after %d iterations (of a maximum %d).\n" ϵᵖʳᵉˢ tol (iter + start_iter) (maxiter + start_iter)
          flush(stdout)
          flush(stderr)
        end
        break
      end

      checkdone(; state=ψ, sweep=start_iter, outputlevel, observer=observer!)
    end
  catch e
    catch_interrupt && isa(e, InterruptException) || rethrow()
    if outputlevel > 0
      @printf "Caught interrupt, stopping VUMPS.\n"
      flush(stdout)
      flush(stderr)
    end
    isnothing(save_func) || save_func(ψ, observer!)
  end
  return ψ, cur_time, ϵᵖʳᵉˢ
end

function vumps_solver(M, time_step, v₀, solver_tol, eager=true)
  λ⃗, v⃗, info = eigsolve(M, v₀, 1, :SR; ishermitian=true, tol=solver_tol, eager)
  return λ⃗[1], v⃗[1], info
end

return function tdvp_solver(M, time_step, v₀, solver_tol, eager=true)
  v, info = exponentiate(M, time_step, v₀; ishermitian=true, tol=solver_tol, eager)
  v = v / norm(v)
  return nothing, v, info
end

function vumps(
  args...;
  time_step=-Inf,
  eigsolve_tol=(x -> x / 100),
  solver_tol=eigsolve_tol,
  eager=true,
  kwargs...,
)
  @assert isinf(time_step) && time_step < 0
  println("Using VUMPS solver with time step $time_step")
  flush(stdout)
  flush(stderr)
  return tdvp(
    vumps_solver, args...; time_step=time_step, solver_tol=solver_tol, eager, kwargs...
  )[1]
end

function ITensorMPS.tdvp(
  args...; time_step, solver_tol=(x -> x / 100), eager=true, outputlevel=0, kwargs...
)
  solver = if !isinf(time_step)
    outputlevel > 0 && println("Using TDVP solver with time step $time_step")
    tdvp_solver
  elseif time_step < 0
    # Call VUMPS instead
    outputlevel > 0 && println("Using VUMPS solver with time step $time_step")
    vumps_solver
  else
    error("Time step $time_step not supported.")
  end
  return tdvp(
    solver,
    args...;
    time_step=time_step,
    solver_tol=solver_tol,
    eager,
    outputlevel,
    kwargs...,
  )
end

"""
  tdvp_subspace_expansion(H, ψ; time_step, outer_iters, inner_iters, outputlevel=0, subspace_expansion_kwargs, tdvp_kwargs)

Run infinite TDVP using the Hamiltonian and MPS supplied.
This will alternate between a bond-by-bond subspace expansion,
and 1-site TDVP.

Once `subspace_expansion_kwargs[:maxdim]` is reached, switch to pure TDVP.

# Arguments
- `H`: The Hamiltonian to use in the time evolution
- `ψ`: The initial state
- `time_step`: Time step for each inner iteration.
- `outer_iters`: Number of times to attempt to expand the bond dimension.
- `inner_iters`: Number of timesteps to take in between expanding the bond dimension
- `outputlevel`: Verbosity level: 1+ for outer loop output, 2+ for TDVP, expansion, & memory usage
- `subspace_expansion_kwargs`: Keyword arguments to pass to `subspace_expansion`
- `tdvp_kwargs`: Keyword arguments to pass to `tdvp`
"""
function itdvp_subspace_expansion(
  H,
  ψ;
  time_step,
  outer_iters,
  inner_iters,
  init_time=0.0,
  outputlevel=0,
  krylov_iters=100,
  subspace_expansion_kwargs,
  tdvp_kwargs,
  save_func=nothing,
)

  cur_time = init_time
  cur_iters = round(Int, cur_time / time_step)
  ITensorMPS.update_observer!(
    tdvp_kwargs[:observer!];
    state=ψ,
    operator=H,
    sweep=cur_iters,
    outputlevel=(outputlevel - 1),
    cur_time,
  )

  prev_krylov_iters = KrylovDefaults.maxiter[]

  try
    KrylovDefaults.maxiter[] = krylov_iters

    N = nsites(ψ)
    bonddims::Union{Nothing, Vector{Int}} = nothing
    prec = Inf
    converged = false
    
    total_time = @elapsed begin
      outer_iter = 0
      while outer_iter < outer_iters
        if outputlevel > 0
          @printf "\nSubspace expansion %d out of %d, starting from dimension %d\n" (outer_iter + 1) outer_iters maxlinkdim(ψ[0:(N + 1)])
          @printf "cutoff = %.1e, maxdim = %d\n" subspace_expansion_kwargs[:cutoff] subspace_expansion_kwargs[:maxdim]
        end

        sub_time = @elapsed ψ = subspace_expansion(
          ψ,
          H;
          outputlevel=(outputlevel - 1),
          subspace_expansion_kwargs...,
        )
        new_bond_dims = linkdims(ψ[0:(N + 1)])

        if outputlevel > 0
          @printf "\nSubspace expansion took %.2f seconds\n" sub_time
          if outputlevel > 1
            println("New bond dimensions: ", new_bond_dims)
          end
        end

        if prec ≤ tdvp_kwargs[:tol] && all(bonddims .== new_bond_dims)
          if outputlevel > 0
            @printf "\niTDVP converged early with precision %.2e and dimension %d\n" prec maximum(new_bond_dims)
          end
          converged = true
          bonddims = new_bond_dims
          break
        end


        # Check if the bond dimension has saturated; TDVP takes some time to set up so we should now run it without stopping.
        if all(new_bond_dims .== subspace_expansion_kwargs[:maxdim])
          bonddims = new_bond_dims
          break
        end

        if outputlevel > 0
          @printf "Running iTDVP with new bond dimension %d\n\n" maximum(new_bond_dims)
        end

        # TODO: add step observers (we have sweep observers)
        tdvp_time = @elapsed ψ, cur_time, prec = tdvp(
          H,
          ψ;
          init_time=cur_time,
          time_step,
          maxiter=inner_iters,
          outputlevel=(outputlevel - 1),
          save_func,
          catch_interrupt=false,
          tdvp_kwargs...,
        )
        bonddims = linkdims(ψ[0:(N + 1)])

        if outputlevel > 0
          @printf "\niTDVP took %.2f seconds\n" tdvp_time
          if outputlevel > 1
            println("New bond dimensions: ", bonddims)
            if outputlevel > 2
              println()
              meminfo_procfs()
            end
          end
        end

        outer_iter += 1
      end

      # TESTME: measure only every n steps to save computation! Need to be careful with consistency in # of steps though.
      if !converged && outer_iter < outer_iters
        remain_iters = (outer_iters - outer_iter) * inner_iters
        if outputlevel > 0
          @printf "\nBond dimension saturated at %d\n" maximum(bonddims)
          @printf "Running iTDVP for %d more iterations\n" remain_iters
        end
        tdvp_time = @elapsed ψ, cur_time, prec = tdvp(
          H,
          ψ;
          init_time=cur_time,
          time_step,
          maxiter=remain_iters,
          outputlevel=(outputlevel - 1),
          save_func,
          tdvp_kwargs...,
        )

        if outputlevel > 0
          prec ≤ tdvp_kwargs[:tol] && @printf "\niTDVP converged with precision %.2e and linkdim %d\n" prec maxlinkdim(ψ[0:(N + 1)])
          @printf "\niTDVP took %.2f seconds\n" tdvp_time
          if outputlevel > 1
            println("Final bond dimensions: ", linkdims(ψ[0:(N + 1)]))
            if outputlevel > 2
              println()
              meminfo_procfs()
            end
          end
        end

      end
    end

    if outputlevel > 0
      @printf "\nTotal time for iTDVP + subspace expansion: %.2f seconds\n" total_time
    end

  catch e
    isa(e, InterruptException) || rethrow(e)
    if outputlevel > 0
      println("\n!!! Caught interrupt, stopping VUMPS !!!")
      flush(stdout)
      flush(stderr)
    end
    isnothing(save_func) || save_func(ψ, tdvp_kwargs[:observer!].data)
  finally
    KrylovDefaults.maxiter[] = prev_krylov_iters
  end

  outputlevel > 0 && println()

  return ψ, cur_time
end
