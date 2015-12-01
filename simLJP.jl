using PyPlot
using PyCall

@pyimport matplotlib.animation as animation

using Distributions

type SimulationSystem
    steps = 2000
    particles = 64
    size = 3.0e-10
    dim = 2
end

type Atom
    diameter = 2.6e-10
    mass = 6.69e-26
end

type LJP
    sigma = 3.4e-10
    epsilon = 1.65e-21
end

type Results
    positons = []
    temperatures = []
end

function ljpCalc(positionA, positionB)
    r = norm(positionA - positionB)
    force = 24 * LJP.epsilon * (2 * (LJP.sigma^12 / r^13) - (LJP.sigma^6 / r^7))
    return force * (positionA - positionB)
end

function adjustPosition(position, velocity, closedBox = true)
    if closedBox == false
        for i = 1 : length(position)
            if position[i] > SimulationSystem.size
                position[i] = -SimulationSystem.size
            elseif position[i] < -SimulationSystem.size
                position[i] = SimulationSystem.size
            end    
        end
    else
        for i = 1 : length(position)
            if position[i] > SimulationSystem.size ||
               position[i] < -SimulationSystem.size
                velocity[i] *= -1
            end
        end     
    end
    return position
end

function temperature(mass, velocities)
    T = 0.0
    kB = 1.38064852e-23

    sum = 0.0
    for velocity in velocities
        for velocityCoord in velocity
            sum += velocityCoord^2
        end
    end
    return sum * mass / (kB * 3 * (SimulationSystem.particles - 1))
end

function simulation()
    println("func:simulation()")

    velocity = sqrt(LJP.epsilon / Atom.mass)
    timeStep = 0.0001 * ljpSigma * sqrt(Atom.mass / LJP.epsilon)
    timeStep2 = timeStep^2

    positions = [[0.0 for i in 1:SimulationSystem.dim]
                 for j in 1:SimulationSystem.particles]
    numPartSide = 64^(1 / SimulationSystem.dim)
    distance = (2 * glSimBoxSize) / (numPartSide + 1)
    for i = 1 : SimulationSystem.particles
        multX = divrem(i, numPartSide)[2]
        multY = divrem(i - 1, numPartSide)[1]

        if multX == 0 multX = numPartSide end

        x = -SimulationSystem.size + distance * multX
        y = -SimulationSystem.size + distance * (multY + 1)
        positions[i] = [x, y]
    end
    
    velocities = [rand(Uniform(-1, 1)) * [1.0, 1.0] * velocity
                  for i in 1:SimulationSystem.particles]
    accelerations = [[0.0, 0.0] for i in 1:SimulationSystem.particles]
    forces = [[0.0, 0.0] for i in 1:SimulationSystem.particles]

    Results.positions = [collect(positions) for i in 1:SimulationSystem.steps]

    # Die Energie des Systems für jeden Zeitschritt.
    finalEnergies = [0.0 for i in 1:SimulationSystem.steps]

    # Die Temperaturen des Systems für jeden Zeitschritt.
    Results.temperatures = [0.0 for i in 1:SimulationSystem.steps]
    
    # Starten der Hauptschleife für die Berechnung der neuen Positionen.
    for i = 2 : SimulationSystem.steps
        for j = 1 : SimulationSystem.particles
            position = Results.positions[i - 1][j]

            # Kräfte aktualisieren...
            force = [0, 0]
            for k = j + 1 : SimulationSystem.particles 
                force = ljpCalc(position, Results.positions[i - 1][k])
                forces[j] += force
                forces[k] += -force
            end

            # Positionen aktualisieren
            newPosition = position + velocities[j] * timeStep + 0.5 *
                           accelerations[j] * timeStep2
            Results.positions[i][j] = adjustPosition(newPosition, velocities[j])

            # Beschleunigungen aktualisieren
            accelerations[j] = forces[j] / Atom.mass

            # Geschwindigkeit aktualisieren
            velocities[j] = velocities[j] + accelerations[j] * timeStep
        end
        # Berechne und speicher die Temperatur für den Schritt.
        Results.temperatures[i] = temperature(Atom.mass, velocities)
        forces = [[0.0, 0.0] for i in 1:SimulationSystem.particles]
    end
end

@time simulation()

fig2 = figure(2)
plot([i for i in 1:SimulationSystem.steps], Results.temperatures)

# function plotSimulation()
#     global finalPositions, simParticles, simSteps, simBoxSize
# 
#     # Die Ergebnisse in einer animation zeichnen.
#     plots = [plot([finalPositions[j][i][1] for i in 1:simParticles],
#                   [finalPositions[j][i][2] for i in 1:simParticles], "ro")
#              for j in 1:simSteps]
# 
#     axis([-simBoxSize, simBoxSize, -simBoxSize, simBoxSize])
#     fig1 = figure(1)
#     animation.ArtistAnimation(fig1, plots, interval = 5, blit = false)
# end
# 
# #plotSimulation()

