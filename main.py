from two_layer import grid
import numpy as np
import time
import multiprocessing
from joblib import Parallel, delayed

import time

def main():
    N           = 50
    n           = 10000
    max_period  = 1000

    geom    = "quad"
    geom    = "hex"
    cond    = 2
    mu      = 0.7000
    alpha   = 1.0020
    beta    = alpha
    k       = 1

    update  = 'asyn'
    update  = 'sync'
	
    lattice = grid(mu, N, N, alpha, beta, geom, cond, update) 

#    lattice.fill_random()
#    tic     = time.perf_counter()
#    lattice.run(n)
#    #lattice.animation(n, n)
#    print(time.perf_counter() - tic )
#    lattice.plot()


#    run     = False
#    run     = True
#
#    add_run = True
#    add_run = False
#
#    if run:
#        lattice.greet()
#        exists  = lattice.load()
#        if not exists:
#            print('Run simulation')
#            lattice.fill_random()
#            lattice.run(n)
#            #lattice.save()
#
#        if add_run:
#            lattice.run(n)
#
#        lattice.plot(save_plot=True)
#    else:
#        lattice.load()
#        lattice.animation(1000, k=k, show_ani=True, save_ani=False)

    mus     = np.arange(0.60, 1.0001, 0.01)
    alphas  = np.arange(1.00, 1.0301, 0.0001)

#    mus     = np.arange(0.60, 1.0001, 0.05)
#    alphas  = np.arange(1.00, 1.0301, 0.001)

#    X       = mus
#    Y       = 1.02*np.ones(mus.size)
    X, Y    = np.meshgrid(mus, alphas)
    parameters  = np.stack((X.flatten(), Y.flatten() ), axis=-1 )
    num_cores   = int(multiprocessing.cpu_count()/2)
    tic = time.perf_counter()
    Parallel(n_jobs=num_cores)(delayed(simulation)(params, n, N, max_period, geom, cond) for params in parameters) 
    print(time.perf_counter() - tic )
#    lattice.periodicity(n, 1000)

#    for mu in np.arange(0.25, 0.7501, .01):
#        lattice = grid(mu, N, N, geom, cond)
#        lattice.greet()
#        lattice.periodicity(n, 1000)

#    lattice.fill_random()
#    lattice.run(n)
#    lattice.save()
#    lattice.plot()


def simulation(params, n, N, max_period, geom, cond):
    mu      = params[0]
    alpha   = params[1]
    beta    = alpha

    lattice = grid(mu, N, N, alpha, beta, geom, cond) 
    lattice.latt_var_period(n, max_period)


if __name__ == "__main__":
    main()
