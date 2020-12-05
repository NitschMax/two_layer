from sandpile import grid
import numpy as np
import time

def main():
    N       = 100
    n       = 100
    crit    = .5
    lattice = grid(crit, N, N)

    lattice.fill_random()
    lattice.run(n)
    lattice.save()
    lattice.load()
    lattice.plot()
    

#    for crit in np.arange(0.19, 0.21, .001):
#        print(crit)
#        lattice = grid(crit, N, N)
#        lattice.fill_random()
#        start   = time.perf_counter()
#        #lattice.load()
#        lattice.run(n)
#        #lattice.plot()
#        print(time.perf_counter() - start)
#        print()
#        lattice.save()


if __name__ == "__main__":
    main()
