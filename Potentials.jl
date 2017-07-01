################################################################################
# Different potentials for molecular dynamics.
################################################################################

module Potentials

type LennardJonesModel
  factor1::Float64
  factor2::Float64
end

"""
Contructor for the LennardJonesModel.
"""
function LennardJonesModel(sigma::Float64, epsilon::Float64)
  factor1::Float64 = epsilon * 24 * 2 * sigma^12
  factor2::Float64 = epsilon * 24 * sigma^6

  return LennardJonesModel(factor1, factor2)
end

"""
Calculating the Lennard-Jones potential force.

m: LennardJonesModel for particles.
r: Distance of particles.
"""
function lennardJonesPotential(m::LennardJonesModel, r::Float64)
  return ((m.factor1 / r^13) - (m.factor2 / r^7))::Float64
end

end
