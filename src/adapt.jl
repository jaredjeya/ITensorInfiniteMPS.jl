
function Adapt.adapt_structure(to, i_mps::InfiniteCanonicalMPS)
  return InfiniteCanonicalMPS(adapt(to, i_mps.AL), adapt(to, i_mps.C), adapt(to, i_mps.AR))
end

function Adapt.adapt_structure(to, i_mps::InfiniteMPS)
  return InfiniteMPS(map(adapt(to), i_mps.data), i_mps.llim, i_mps.rlim, i_mps.reverse)
end

function Adapt.adapt_structure(to, i_mpo::InfiniteSum)
    return InfiniteSum(adapt(to, i_mpo.data))
end

function Adapt.adapt_structure(to, i_mpo::CelledVector)
  return CelledVector(map(adapt(to), i_mpo.data), i_mpo.translator)
end

ITensors.datatype(i_mps::InfiniteCanonicalMPS) = ITensors.datatype(i_mps.AL)
ITensors.datatype(i_mps::InfiniteMPS) = ITensors.datatype(i_mps[1])
ITensors.datatype(i_mpo::InfiniteSum) = ITensors.datatype(i_mpo.data[1])