from two_layer import grid
import numpy as np
import time

def main():
    N       = 100
    n       = 10000
    mu      = 0.50
    geom    = "hex"
    geom    = "quad"
    cond    = 0
	

    lattice = grid(mu, N, N, geom, cond) 
    for mu in np.arange(0.46, 0.8501, .01):
        lattice = grid(mu, N, N, geom, cond)
        lattice.greet()
        lattice.periodicity(n, 1000)

#    lattice.fill_random()
#    lattice.run(n)
#    lattice.save()
#    lattice.load()
#    lattice.plot()
#
#    lattice.fill_random()
#    lattice.animation(n)
 
#    for mu in np.arange(0.5, 4.001, .1):
#        lattice = grid(mu, N, N, geom, cond)
#        lattice.greet()
#        for i in range(10):
#            print(i)
#            lattice.fill_random()
#            lattice.run(n)
#            lattice.save()
# 
#
#

if __name__ == "__main__":
    main()
