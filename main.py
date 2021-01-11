from two_layer import grid
import numpy as np
import time

def main():
    N       = 100
    n       = 300
    mu      = .50
    geom    = "hex"
    geom    = "quad"
    cond    = 0
	

    lattice = grid(mu, N, N, geom, cond)
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
#        lattice = grid(mu, N, N, 'quad', 0)
#        lattice.greet()
#        lattice.fill_random()
#        lattice.run(n)
#        lattice.save()
# 
if __name__ == "__main__":
    main()
