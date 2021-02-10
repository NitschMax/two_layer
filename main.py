from two_layer import grid
import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed

def main():
    N       = 100
    n       = 10000
    geom    = "hex"
    geom    = "quad"
    cond    = 2
    mu      = 0.800
    alpha   = 1.015
    beta    = alpha
	
    run     = False
    run     = True

    lattice = grid(mu, N, N, alpha, beta, geom, cond) 

    if run:
        lattice.greet()
        exists  = lattice.load()
        if not exists:
            print('Run simulation')
            lattice.fill_random()
            lattice.run(n)
            lattice.save()

        lattice.plot(save_plot=False)
    else:
        lattice.load()
        lattice.fill_random()
        lattice.animation(500, k=100, show_ani=True, save_ani=False)

#    mus     = np.arange(.5, 2.3001, 0.01)
#    alphas  = np.arange(1.00, 1.080, 0.01)
#    X       = mus
#    Y       = 1.02*np.ones(mus.size)
##    X, Y    = np.meshgrid(mus, alphas)
#    parameters  = np.stack((X.flatten(), Y.flatten() ), axis=-1 )
#    num_cores   = multiprocessing.cpu_count()
#    Parallel(n_jobs=num_cores)(delayed(simulation)(params, n, N, geom, cond) for params in parameters) 
##    lattice.periodicity(n, 1000)

#    for mu in np.arange(0.25, 0.7501, .01):
#        lattice = grid(mu, N, N, geom, cond)
#        lattice.greet()
#        lattice.periodicity(n, 1000)

#    lattice.fill_random()
#    lattice.run(n)
#    lattice.save()
#    lattice.plot()


def simulation(params, n, N, geom, cond):
    mu      = params[0]
    alpha   = params[1]
    beta    = alpha

    lattice = grid(mu, N, N, alpha, beta, geom, cond) 
    lattice.greet()
    lattice.fill_random()
    lattice.run(n)
    lattice.save()


if __name__ == "__main__":
    main()
