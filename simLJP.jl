#!/usr/bin/env julia

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
  dim::Int64
  initTemp::Float64

  # Physical properties of the particles.
  diameter::Float64
  mass::Float64

  # Properties for the Lennard-Jones-Potential.
  sigma::Float64
  epsilon::Float64

  timeStep::Float64
  timeStep2::Float64
end

################################################################################
# Functions

"""
Constructor function for type Model.

return: The contructed model object.
"""
function Model()
  steps::Int64 = 5000
  particles::Int64 = 64
  sideLength::Float64 = 6.0e-10
  dim::Int64 = 2
  initTemp::Float64 = 25
  diameter::Float64 = 2.6e-10
  mass::Float64 = 6.69e-26
  sigma::Float64 = 3.4e-10
  epsilon::Float64 = 1.65e-21
  timeStep::Float64 = 0.0001 * sigma * sqrt(mass / epsilon)
  timeStep2::Float64 = timeStep^2

  return Model(steps, particles, sideLength, dim, initTemp, diameter, mass,
               sigma, epsilon, timeStep, timeStep2)
end

"""
Calculating the force between two particles based on the
Lennard-Jones-Potential.

positionA: Position of the first particle.
positionB: Position of the second particle. 

return:    A vector with the resulting forces to every coordinate.
"""
function calculateLJP(positionA::Array{Float64, 1},
                      positionB::Array{Float64, 1}, epsilon::Float64,
                      sigma::Float64)
  r::Float64 = norm(positionA - positionB)
  force::Float64 = 24 * epsilon * (2 * (sigma^12 / r^13) - (sigma^6 / r^7))

  return force * (positionA - positionB)
end

"""
Adjust the given position to fit in the given area. There may be a closed or
a periodic box.

position:     The positions to correct.
velocity:     The current velocity of the particle.
sideLength:   The length of a box with equal side length.
systemClosed: True if the system is a closed box and false if it is periodic. 

return:       The new position and velocity as a two component list.
"""
function adjustPosition(position::Array{Float64, 1}, dim::Int64,
                        velocity::Array{Float64, 1}, sideLength::Float64,
                        systemClosed::Bool = true)
  # We suggest, that the center of the box has the coordinates [0, 0, 0].
  halfLength::Float64 = sideLength / 2.0
  
  if systemClosed::Bool == false
    for i::Int64 = 1 : dim
      if position[i] > halfLength
        position[i] = -halfLength
      elseif position[i] < -halfLength
        position[i] = halfLengh
      end    
    end
  else
    for i::Int64 = 1 : dim
      if position[i] > halfLength || position[i] < -halfLength
        velocity[i] *= -1
      end
    end
  end

  return (position, velocity)
end

"""
Calculate the temperature of the system based on the mass and velocity of
every particle.

mass:     Mass of the particle.
velocity: Velocity of the particle as a vector with n-coordinates.

return:   The temperature in unit Kelvin.
"""
function temperature(mass::Float64, velocities::Array{Float64, 2},
                     particles::Int64)
  const kB::Float64 = 1.38064852e-23
  
  s::Float64 = 0.0
  for i::Int64 = 1:size(velocities, 1), j::Int64 = 1:size(velocities, 2)
    s += velocities[i, j]^2
  end

  return s * mass / (kB * 3.0 * (particles - 1))
end

"Main simulation function for running the system."
function simulation()
  m::Model = Model()
  
  # Create the positions array and init the start positions.
  positions::Array{Float64, 3} = fill(0.0, m.dim, m.particles, m.steps)

  numPartSide::Float64 = 64.0^(1.0 / m.dim)
  distance::Float64 = m.sideLength / (numPartSide + 1.0)
  multX::Float64 = 0.0
  multY::Float64 = 0.0
  for i::Int64 = 1 : m.particles
    multX = divrem(i, numPartSide)[2]
    multY = divrem(i - 1, numPartSide)[1]
    
    if multX == 0 multX = numPartSide end

    positions[1, i, 1] = -(m.sideLength / 2.0) + distance * multX
    positions[2, i, 1] = -(m.sideLength / 2.0) + distance * (multY + 1.0)
  end

  # Create the velocity array and init the coordinates to the normal
  # distribution.
  dist = Normal(0.0, m.initTemp^(0.5))
  velocities::Array{Float64, 2} = rand(dist, m.dim, m.particles)

  # Create the accelerations, temperatures and forces array, which are zero at
  # the beginning.
  accelerations::Array{Float64, 2} = fill(0.0, m.dim, m.particles)
  forces::Array{Float64, 2} = fill(0.0, m.dim, m.particles)
  temperatures::Array{Float64, 1} = fill(0.0, m.steps)

  positionA::Array{Float64, 1} = fill(0.0, m.dim)
  positionB::Array{Float64, 1} = fill(0.0, m.dim)
  velocity::Array{Float64, 1} = fill(0.0, m.dim)
  acceleration::Array{Float64, 1} = fill(0.0, m.dim)
  # Running main loop
  for i::Int64 = 2 : m.steps
    for j::Int64 = 1 : m.particles
      positionA[1] = positions[1, j, i - 1]
      positionA[2] = positions[2, j, i - 1]
      
      # Update forces
      for k::Int64 = j + 1 : m.particles
        positionB[1] = positions[1, k, i - 1]
        positionB[2] = positions[2, k, i - 1]
        
        #forces[:, j] += calculateLJP(positionA, positionB, m.epsilon, m.sigma)
        force = calculateLJP(positionA, positionB, m.epsilon, m.sigma)
        forces[1, j] += force[1]
        forces[2, j] += force[2]
        forces[1, k] -= force[1]
        forces[2, k] -= force[2]
      end

      # Update particle position followed by a correction of particle position
      # and velocitiy.
      velocity[1] = velocities[1, j]
      velocity[2] = velocities[2, j]
      acceleration[1] = accelerations[1, j]
      acceleration[2] = accelerations[2, j]
      result = adjustPosition(positionA + velocity * m.timeStep + 0.5 *
                              acceleration * m.timeStep2, m.dim,
                              velocity, m.sideLength)
      positions[1, j, i] = result[1][1]
      positions[2, j, i] = result[1][2]
      velocities[1, j] = result[2][1]
      velocities[2, j] = result[2][2]

      # Update accelerations
      accelerations[1, j] = forces[1, j] / m.mass
      accelerations[2, j] = forces[2, j] / m.mass

      # Update velocities
      velocities[1, j] = velocities[1, j] + accelerations[1, j] * m.timeStep
      velocities[2, j] = velocities[2, j] + accelerations[2, j] * m.timeStep
    end
    
    # Calculate and save temperature for every step.
    temperatures[i] = temperature(m.mass, velocities, m.particles)
    fill!(forces, 0.0)
  end
  
  return (positions, temperatures)
end

#@code_warntype simulation()
@time result = simulation()

function showAnimationPlot()
  positions = result[1]
  plots = [pyplot.plot([positions[1, i, j] for i in 1:size(positions, 2)],
                [positions[2, i, j] for i in 1:size(positions, 2)], "ro")
           for j in 1:size(positions, 3)]
  
  pyplot.axis([-6e-10, 6e-10, -6e-10, 6e-10])
  fig1 = pyplot.figure(1)
  animation.ArtistAnimation(fig1, plots, interval = 1, blit = false)
  pyplot.show()
end

function showTemperaturePlot()
  pyplot.title("Molecular Dynamics Simulation", fontsize = 14)
  pyplot.plot([i for i in 1:size(result[2], 1)], result[2])
  pyplot.xlabel("Timesteps")
  pyplot.ylabel("Temperature [K]")
  pyplot.show()
end

showTemperaturePlot()
# showAnimationPlot()

