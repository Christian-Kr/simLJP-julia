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
# Functions

"""
Calculating the force between two particles based on the
Lennard-Jones-Potential.

positionA: Position of the first particle.
positionB: Position of the second particle. 

return:    A vector with the resulting forces to every coordinate.
"""
function calculateLJP(positionA::Vector, positionB::Vector, epsilon::Float64,
                      sigma::Float64)
  r = norm(positionA - positionB)
  force = 24 * epsilon * (2 * (sigma^12 / r^13) - (sigma^6 / r^7))

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
function adjustPosition(position::Vector, velocity::Vector, sideLength::Float64,
                        systemClosed = true)
  # We suggest, that the center of the box has the coordinates [0, 0, 0].
  halfLength = sideLength / 2
  
  if systemClosed == false
    for i = 1 : length(position)
      if position[i] > halfLength
        position[i] = -halfLength
      elseif position[i] < -halfLength
        position[i] = halfLengh
      end    
    end
  else
    for i = 1 : length(position)
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
                     particles::Int32)
  const kB::Float64 = 1.38064852e-23
  
  s = 0.0
  for i = 1:size(velocities, 1), j = 1:size(velocities, 2)
    s += velocities[i, j]^2
  end

  return s * mass / (kB * 3 * (particles - 1))
end

"Main simulation function for running the system."
function simulation()
  # Different properties for the simulated system.
  const steps::Int32 = 5000
  const particles::Int32 = 64
  const sideLength::Float64 = 6.0e-10
  const dim::Int32 = 2
  const initTemp::Int32 = 25

  # Physical properties of the particles.
  const diameter::Float64 = 2.6e-10
  const mass::Float64 = 6.69e-26

  # Properties for the Lennard-Jones-Potential.
  const sigma::Float64 = 3.4e-10
  const epsilon::Float64 = 1.65e-21

  timeStep::Float64 = 0.0001 * sigma * sqrt(mass / epsilon)
  timeStep2::Float64 = timeStep^2

  # Create the positions array and init the start positions.
  positions::Array{Float64, 3} = fill(0.0, dim, particles, steps)

  numPartSide = 64^(1 / dim)
  distance = sideLength / (numPartSide + 1)
  multX::Float64 = 0.0
  multY::Float64 = 0.0
  for i = 1 : particles
    multX = divrem(i, numPartSide)[2]
    multY = divrem(i - 1, numPartSide)[1]
    
    if multX == 0 multX = numPartSide end

    positions[:, i, 1] = [-(sideLength / 2) + distance * multX,
      -(sideLength / 2) + distance * (multY + 1)]
  end

  # Create the velocity array and init the coordinates to the normal
  # distribution.
  dist = Normal(0.0, sqrt(initTemp))
  velocities::Array{Float64, 2} = fill(0.0, dim, particles)
  for i = 1:dim, j = 1:particles
    velocities[i, j] = rand(dist)
  end

  # Create the accelerations, temperatures and forces array, which are zero at
  # the beginning.
  accelerations::Array{Float64, 2} = fill(0.0, dim, particles)
  forces::Array{Float64, 2} = fill(0.0, dim, particles)
  temperatures::Array{Float64, 1} = fill(0.0, steps)
    
  # Running main loop
  for i = 2 : steps
    for j = 1 : particles
      position = positions[:, j, i - 1]
      
      # Update forces
      force::Vector = [0.0, 0.0]
      for k = j + 1 : particles 
        force = calculateLJP(position, positions[:, k, i - 1], epsilon, sigma)
        forces[:, j] += force
        forces[:, k] += -force
      end

      # Update particle position followed by a correction of particle position
      # and velocitiy.
      positions[:, j, i], velocities[:, j] = adjustPosition(position +
        velocities[:, j] * timeStep + 0.5 * accelerations[:, j] * timeStep2,
        velocities[:, j], sideLength)

      # Update accelerations
      accelerations[:, j] = forces[:, j] / mass

      # Update velocities
      velocities[:, j] = velocities[:, j] + accelerations[:, j] * timeStep
    end
    
    # Calculate and save temperature for every step.
    temperatures[i] = temperature(mass, velocities, particles)
    fill!(forces, 0.0)
  end
  
  return (positions, temperatures)
end

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

