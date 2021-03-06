import data_dir
import os
from two_layer import grid
import matplotlib.pyplot as plt
import numpy as np

def main():
    N           = 100
    geom        = "hex"
    geom        = "quad"
    cond        = 0
    mu          = 1.0
    alpha       = 1.2
    beta        = alpha

    lattice     = grid(mu, N, N, alpha, beta, geom, cond)
    fig, ax1    = plt.subplots()
    ax2         = ax1.twinx()

    mu_var      = np.arange(0.3, 1.5, 0.01)
    alpha_var   = alpha*np.ones(mu_var.size )
    beta_var    = beta*np.ones(mu_var.size )
    variables   = np.transpose([mu_var, alpha_var, beta_var])

    #period_eval(lattice, ax1, ax2)
    var_eval(lattice, ax1, variables)
    plt.show()


def period_eval(lattice, ax1, ax2):
    old_dir     = os.getcwd()
    directory   = lattice.data_directory()
    dir         = 'periodicities/' + directory[1] + directory[2]
    files       = os.listdir(dir)
    os.chdir(dir)
    files       = filter(os.path.isfile, files)
    period      = np.array([np.load(file) for file in files]).reshape(-1, 2)
    period      = period[period[:,0].argsort() ]
    mu          = period[:,0]
    period      = period[:,1]

    per         = np.where(period != False )[0]
    non_per     = np.where(period == False )[0]

    bins        = np.arange(mu[non_per[0] ]-.005, mu[non_per[-1] ]+.005, .01)
    ax2.hist(mu[non_per], bins=bins, color='gray' )
    ax1.scatter(mu[per], period[per], marker='x')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Average filling mu')
    ax1.set_ylabel('Periodicity')
    ax2.set_ylabel('Number of configurations without periodicity')
    ax1.grid(True)
    plt.savefig('plots/periodicities.pdf')
    os.chdir(old_dir)

def var_in(word):
    return 'var' in word

def var_eval(lattice, ax, variables):
    directory   = lattice.data_directory()
    latt_dir    = directory[0] + directory[1] + directory[2]
    var         = []

    for set_of_var in variables:
        lattice.mu     = set_of_var[0]
        lattice.alpha  = set_of_var[1]
        lattice.beta   = set_of_var[2]

        mu_direct   = lattice.data_clarification()
        if not os.path.isdir(latt_dir + mu_direct):
            continue

        files   = os.listdir(latt_dir + mu_direct )
        files   = list(filter(var_in, files) )

        data    = np.array([ np.concatenate( (set_of_var, np.load(latt_dir + mu_direct + "/" + datei)[-1]) ) for datei in [files[0]] ] )
        var.append(data)

    var         = np.array(var)
    var         = var.reshape(-1,5)
    var         = var[var[:,0].argsort() ]
    mu          = var[:, 0]
    var_down    = var[:, 3]
    var_down_n  = np.sqrt(var_down)/mu
    var_upp     = var[:, 4]
    var_upp_n   = np.sqrt(var_upp)/mu

    var_dir     = "variances/" + directory[1] + directory[2]
    if not os.path.exists(var_dir):
        os.mkdir(var_dir)
    
    ax.scatter(mu, var_upp_n, marker='.', label='upper layer' )
    ax.scatter(mu, var_down_n, marker='.', label='lower layer')
    ax.grid(True)
    ax.set_xlabel("mu")
    ax.set_ylabel("normed standard deviation")
    np.savetxt(var_dir + lattice.geom + '_variance_phases.txt', np.transpose([mu, var_down_n, var_upp_n] ) )
    ax.legend()

    plt.savefig(var_dir + lattice.geom + '_variance_phases.pdf')
    

if __name__ == "__main__":
    main()
