from two_layer import grid
import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed

def main():
    N       = 100
    n       = 4000
    geom    = "hex"
    geom    = "quad"
    cond    = 0
    mu      = 0.40
    alpha   = 1.20
    beta    = alpha
	

    lattice = grid(mu, N, N, alpha, beta, geom, cond) 
    lattice.greet()

    lattice.fill_random()
    lattice.run(n)
    print('Average moisture in the upper layer', np.mean(lattice.upp) )
    print('Average moisture in the lower layer', np.mean(lattice.down) )
    lattice.plot()

#    mus     = np.arange(.3, 1.5001, 0.01)
#    alphas  = np.arange(2.0, 2.001, 0.1)
#    X, Y    = np.meshgrid(mus, alphas)
#    parameters  = np.stack((X.flatten(), Y.flatten() ), axis=-1 )
#    num_cores   = multiprocessing.cpu_count()
#    Parallel(n_jobs=num_cores)(delayed(simulation)(params, n, N, geom, cond) for params in parameters) 
#    lattice.periodicity(n, 1000)

#    for mu in np.arange(0.25, 0.7501, .01):
#        lattice = grid(mu, N, N, geom, cond)
#        lattice.greet()
#        lattice.periodicity(n, 1000)

#    lattice.fill_random()
#    lattice.run(n)
#    lattice.save()
#    lattice.plot()

    lattice.fill_random()
    lattice.animation(n)
 
#    for alpha in np.arange(1.1, 2.000, .1):
#        beta    = alpha
#        for mu in np.arange(0.30, 1.501, .1):
#            lattice = grid(mu, N, N, alpha, beta, geom, cond) 
#            lattice.greet()
#            for i in range(1):
#                #print(i)
#                lattice.fill_random()
#                lattice.run(n)
#                lattice.save()
 

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
