import numpy as np
import matplotlib.pyplot as plt

G = 6.6743e-11 # m^3 kg^-1 s^-2

massLo = 1e1
massHi = 1e8

xlim = [-5, 5]
ylim = [-5, 5]

def initMass(low, high):
    return low + (high - low) * np.random.rand()

def initPos(xlim, ylim):
    return np.array([xlim[0] + (xlim[1] - xlim[0]) * np.random.rand(), ylim[0] + (ylim[1] - ylim[0]) * np.random.rand()])

def normalizePos(pos):

    if pos[0] < xlim[0]:
        pos[0] = xlim[1] - np.random.rand()
    if pos[0] > xlim[1]:
        pos[0] = xlim[0] + np.random.rand()
    if pos[1] < ylim[0]:
        pos[1] = ylim[1] - np.random.rand()
    if pos[1] > ylim[1]:
        pos[1] = ylim[0] + np.random.rand()

    return pos

def force(p, q): # Force applied on p exerted by q
    mass = p.mass * q.mass
    d = p.pos - q.pos
    norm = np.linalg.norm(d)
    return -G * (mass / norm**2) * (d / norm)

class Particle:

    def __init__(self, m, pos):

        self.mass = m               # [kg]
        self.pos = pos
        self.vel = np.zeros(len(pos))
        self.acc = np.zeros(len(pos))

    def update(self, F, dt=1.): # F is a vector with unit [kg*m/s^2]

        self.acc = F / self.mass
        self.vel = self.vel + self.acc * dt
        self.pos = self.pos + self.vel * dt

def updateParticles(particles):

    # Copy particles
    old = particles.copy()

    # Update particles
    for i in range(len(old)):

        acc = np.zeros(2)

        # Calculate force acting on current particle
        for j in range(len(old)):
            if i != j:
                d = old[j].pos - old[i].pos
                distSq = np.linalg.norm(d)**2
                f = (G * old[j].mass) / (distSq * np.sqrt(distSq + .15))
                acc += d * f

        # Update current particle with current force
        #particles[i].update(F, 1.)
        dt = 1.
        particles[i].vel = particles[i].vel + acc * dt
        particles[i].pos = particles[i].pos + particles[i].vel * dt

        # Restrict position
        particles[i].pos = normalizePos(particles[i].pos)

    return particles

np.random.seed(42)
N = 50
particles = []
for i in range(N):
    particles.append(Particle(initMass(massLo, massHi), initPos(xlim, ylim)))
maxMass = sum([p.mass for p in particles]) *.25

while True:

    particles = updateParticles(particles)

    sizes = [(1 + (p.mass - massLo) / (massHi - massLo))**10 for p in particles]
    plt.scatter([p.pos[0] for p in particles], [p.pos[1] for p in particles], c=np.arange(N), s=sizes)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

    # Create image
    #img = np.zeros((xlim[1] - xlim[0], ylim[1] - ylim[0]))
    #for i in range(xlim[1] - xlim[0]):
    #    for j in range(ylim[1] - ylim[0]):
    #        x = -5 + i
    #        y = -5 + j
    #        hx = x + 1
    #        hy = y + 1
    #        mass = sum([p.mass for p in particles if p.pos[0] >= x and p.pos[0] < hx and p.pos[1] >= y and p.pos[1] < hy])
    #        img[i][j] = mass

    #plt.subplot(2,1,2)
    #plt.matshow(img, vmin=0., vmax=maxMass)
    #plt.show()

