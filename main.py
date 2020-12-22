from two_layer import grid
import numpy as np
import time

def main():
    N       = 100
    n       = 2000
    mu      = 1.50
    geom    = "hex"
    geom    = "quad"

    lattice = grid(mu, N, N, geom)
    lattice.greet()

    lattice.fill_random()
    lattice.run(n)
    lattice.save()
    lattice.load()
    lattice.plot()
#
#    lattice.fill_random()
#    lattice.animation(n)
#
#    for mu in np.arange(0.1, 4.001, .1):
#        lattice = grid(mu, N, N, 'quad')
#        lattice.greet()
#        lattice.fill_random()
#        lattice.run(n)
#        lattice.save()
# 
if __name__ == "__main__":
    main()
