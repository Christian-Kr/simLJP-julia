#!/usr/bin/env julia
################################################################################
# This script is a short program for everyone who is interested in molecular
# dynamics. It will simulate a small amount of gas (default is argon) in a fixed
# volume. The potential will be desribed by the Lennard-Jones potential. The
# simulation itself is implemented with the velocity Verlet algorithm.
# Feel free to change/distribute/whatever the script.
# Also, if you have questions feel free to contact me at:
#     <Coding ät Christian-Krippendorf.de>
################################################################################

using PyCall

@pyimport matplotlib.pyplot as pyplot
@pyimport matplotlib.animation as animation

using Distributions

################################################################################
# Global variables
# Removed most global variables cause of performance issues as explained in:
# http://docs.julialang.org/en/release-0.4/manual/performance-tips/

################################################################################
# Types

type Model
  # Different properties for the simulated system.
  steps::Int64
  particles::Int64
  sideLength::Float64
  halfSideLength::Float64
  initTemp::Float64

  # Physical properties of the particles.
  diameter::Float64
  mass::Float64

  # Properties for the Lennard-Jones-Potential.
  sigma::Float64
  sigma6::Float64
  sigma12::Float64
  epsilon::Float64
  epsilon24::Float64
  factor1::Float64
  factor2::Float64
  
  timeStep::Float64
  timeStep2::Float64

  rcut::Float64
  
  positions::Array{Float64, 3}
  velocities::Array{Float64, 2}
 
  accelerations::Array{Float64, 2}
  forces::Array{Float64, 2}
  temperatures::Array{Float64, 1}
end

################################################################################
# Functions

function correctVelocities!(velocities::Array{Float64, 2}, mass::Float64,
    particles::Int64, initTemp::Int64)

  s::Float64 = 0.0
  for i::Int64 = 1:size(velocities, 1), j::Int64 = 1:size(velocities, 2)
    s += velocities[i, j]^2
  end

  lambda = ((3 * (particles - 1) * 1.38064852e-23 * initTemp) / (mass * s))^(1/2)

  for i::Int64 = 1:size(velocities, 1), j::Int64 = 1:size(velocities, 2)
    velocities[i, j] *= lambda
  end
end

function initPositions!(positions::Array{Float64, 3}, sideLength::Float64,
      halfSideLength::Float64, particles::Int64)

  # How many particles on a line.
  numPartSide::Int64 = Int64(round(particles^(1.0 / 3.0)))
  numPartSide2::Int64 = numPartSide^2
  
  #The distance between every particle and the box limit.
  distance::Float64 = sideLength / (numPartSide + 1.0)

  parPos::Int64 = 0
  for x = 1:numPartSide, y = 1:numPartSide, z = 1:numPartSide
    parPos = z + (y - 1) * numPartSide + (x - 1) * numPartSide^2
    positions[1, parPos, 1] = -halfSideLength + distance * z
    positions[2, parPos, 1] = -halfSideLength + distance * y
    positions[3, parPos, 1] = -halfSideLength + distance * x
  end
end

"""
Constructor function for type Model.

return: The contructed model object.
"""
function Model()
  # To high takes to long. :-D
  steps::Int64 = 10000

  # Please choose a number with an integer result at ^(1/3).
  particles::Int64 = 64

  # The side length is one complete side length of the box.
  sideLength::Float64 = 6.0e-10
  halfSideLength::Float64 = sideLength / 2.0

  # The starting temperature of the particles which correlates to the velocity
  # of the particles.
  initTemp::Int64 = 50

  # Properties related to the atoms/molecules you choose.
  diameter::Float64 = 2.6e-10
  mass::Float64 = 6.69e-26

  # Different properties for the Lennard-Jones potential. You should change them
  # only if you know hat you are doing as they are well choosen to the default
  # system.
  sigma::Float64 = 3.4e-10
  sigma6::Float64 = sigma^6
  sigma12::Float64 = sigma^12
  epsilon::Float64 = 1.65e-21
  epsilon24::Float64 = 24 * epsilon
  factor1::Float64 = epsilon24 * 2 * sigma12
  factor2::Float64 = epsilon24 * sigma6

  # Properties of the simulated system.
  timeStep::Float64 = 0.001 * sigma * sqrt(mass / epsilon)
  timeStep2::Float64 = timeStep^2
  rcut::Float64 = 2 * sigma

  # Init the initial positions.
  positions::Array{Float64, 3} = fill(0.0, 3, particles, steps)
  initPositions!(positions, sideLength, halfSideLength, particles)

  # Create the velocity array and init the coordinates to the normal
  # distribution.
  dist = Normal(0.0, initTemp^(0.5))
  velocities::Array{Float64, 2} = rand(dist, 3, particles)
  correctVelocities!(velocities, mass, particles, initTemp);

  # Create the accelerations, temperatures and forces array, which are zero at
  # the beginning.
  accelerations::Array{Float64, 2} = fill(0.0, 3, particles)
  forces::Array{Float64, 2} = fill(0.0, 3, particles)
  temperatures::Array{Float64, 1} = fill(0.0, steps)

  return Model(steps, particles, sideLength, halfSideLength, initTemp,
               diameter, mass, sigma, sigma6, sigma12, epsilon, epsilon24,
               factor1, factor2, timeStep, timeStep2, rcut, positions,
               velocities, accelerations, forces, temperatures)
end

"""
Calculating the Lennard-Jones potential force.

m: Model of simulation.
r: Distance of particles.
"""
function ljp(m::Model, r::Float64)
  return ((m.factor1 / r^13) - (m.factor2 / r^7))::Float64
end

"""
Calculating the force between two particles.

posIndexA: Index of the first particle.
posIndexB: Index of the second particle.
step:      Step of simulation.
"""
function potential!(m::Model, posIndexA::Int64, posIndexB::Int64, step::Int64)
  diff::Array{Float64, 1} = fill(0.0, 3)
  diff[1] = m.positions[1, posIndexA, step] - m.positions[1, posIndexB, step]
  diff[2] = m.positions[2, posIndexA, step] - m.positions[2, posIndexB, step]
  diff[3] = m.positions[3, posIndexA, step] - m.positions[3, posIndexB, step]
  
  r::Float64 = norm(diff)
  if r < m.rcut
    pot::Float64 = ljp(m, r)
  
    m.forces[1, posIndexA] += pot * diff[1]
    m.forces[2, posIndexA] += pot * diff[2]
    m.forces[3, posIndexA] += pot * diff[3]
    m.forces[1, posIndexB] -= pot * diff[1]
    m.forces[2, posIndexB] -= pot * diff[2]
    m.forces[3, posIndexB] -= pot * diff[3]
  end

  return 0.0
end

"""
Adjust the given position to fit in the given area. There may be a closed or
a periodic box.

position:     The positions to correct.
velocity:     The current velocity of the particle.
sideLength:   The length of a box with equal side length.
systemClosed: True if the system is a closed box and false if it is periodic. 
"""
function adjustPosition!(position::Array{Float64, 1},
    velocities::Array{Float64, 2}, index::Int64, halfSideLength::Float64,
    systemClosed::Bool = true)

  # We suggest, that the center of the box has the coordinates [0, 0, 0].
  if systemClosed::Bool == false
    for i::Int64 = 1 : 3
      if position[i] > halfSideLength
        position[i] = -halfSideLength
      elseif position[i] < -halfSideLength
        position[i] = halfSideLengh
      end    
    end
  else
    for i::Int64 = 1 : 3
      if position[i] > halfSideLength || position[i] < -halfSideLength
        velocities[i, index] *= -1
      end
    end
  end
end

"""
Calculate the temperature of the system based on the mass and velocity of
every particle.

temperatures: The temperatures object.
index:        The index for temperature step.
mass:         Mass of the particle.
velocities:   Velocities object of all particle as a vector with n-coordinates.
"""
function temperature!(temperatures::Array{Float64, 1}, index::Int64,
    mass::Float64, velocities::Array{Float64, 2}, particles::Int64)

  const kB::Float64 = 1.38064852e-23
  s::Float64 = 0.0
  for i::Int64 = 1:size(velocities, 1), j::Int64 = 1:size(velocities, 2)
    s += velocities[i, j]^2
  end

  temperatures[index] =  s * mass / (kB * 3.0 * (particles - 1))
end

"Main simulation function for running the system."
function simulation(m::Model)
  position::Array{Float64, 1} = fill(0.0, 3)

  # Update forces
  for j = 1:m.particles
    for k = j + 1:m.particles
      potential!(m, j, k, 1)
    end

    # Update accelerations
    m.accelerations[1, j] = m.forces[1, j] / m.mass
    m.accelerations[2, j] = m.forces[2, j] / m.mass
    m.accelerations[3, j] = m.forces[3, j] / m.mass
  end
  
  # Running main loop
  for i = 1 : m.steps - 1
    for j = 1 : m.particles
      # Update velocities + 0.5 timestep.
      m.velocities[1, j] = m.velocities[1, j] + 0.5 * m.accelerations[1, j] * m.timeStep
      m.velocities[2, j] = m.velocities[2, j] + 0.5 * m.accelerations[2, j] * m.timeStep
      m.velocities[3, j] = m.velocities[3, j] + 0.5 * m.accelerations[3, j] * m.timeStep

      # Update particle position followed by a correction of particle position
      # and velocitiy.
      position[1] = m.positions[1, j, i] + m.velocities[1, j] * m.timeStep
      position[2] = m.positions[2, j, i] + m.velocities[2, j] * m.timeStep
      position[3] = m.positions[3, j, i] + m.velocities[3, j] * m.timeStep      
      adjustPosition!(position, m.velocities, j, m.halfSideLength)
      m.positions[1, j, i + 1] = position[1]
      m.positions[2, j, i + 1] = position[2]
      m.positions[3, j, i + 1] = position[3]      
    end
    for j = 1 : m.particles
      # Update forces
      for k = j + 1 : m.particles
        potential!(m, j, k, i + 1)
      end
      
      # Update accelerations
      m.accelerations[1, j] = m.forces[1, j] / m.mass
      m.accelerations[2, j] = m.forces[2, j] / m.mass
      m.accelerations[3, j] = m.forces[3, j] / m.mass

      # Update velocities + full timestep.
      m.velocities[1, j] = m.velocities[1, j] + 0.5 * m.accelerations[1, j] * m.timeStep
      m.velocities[2, j] = m.velocities[2, j] + 0.5 * m.accelerations[2, j] * m.timeStep
      m.velocities[3, j] = m.velocities[3, j] + 0.5 * m.accelerations[3, j] * m.timeStep
    end
    
    # Calculate and save temperature for every step.
    temperature!(m.temperatures, i, m.mass, m.velocities, m.particles)
    fill!(m.forces, 0.0)
  end
end

function showAnimationPlot(m::Model)
  plots = [pyplot.plot([m.positions[1, i, j] for i in 1:size(m.positions, 2)],
                [m.positions[2, i, j] for i in 1:size(m.positions, 2)], "ro")
           for j in 1:size(m.positions, 3)]
  
  pyplot.axis([-6e-10, 6e-10, -6e-10, 6e-10])
  fig1 = pyplot.figure(1)
  animation.ArtistAnimation(fig1, plots, interval = 1, blit = false)
  pyplot.show()
end

function showTemperaturePlot(m::Model)
  pyplot.title("Molecular Dynamics Simulation", fontsize = 14)
  pyplot.plot([i for i in 1:m.steps], m.temperatures)
  pyplot.axis([0, m.steps, m.initTemp - 20, m.initTemp + 20])
  pyplot.xlabel("Timesteps")
  pyplot.ylabel("Temperature [K]")
  pyplot.show()
end

function writeAnimation(m::Model)
  for i = 1:m.steps
    file = open(string("result/sim-", i, ".csv"), "w")

    for j = 1:m.particles
      write(file, join(m.positions[:, j, i], ","), "\n")
    end

    close(file)
  end
end

m = Model()
@time simulation(m)

showTemperaturePlot(m)

