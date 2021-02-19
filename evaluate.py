import data_dir
import os
from two_layer import grid
import matplotlib.pyplot as plt
import numpy as np

def main():
    N           = 100
    geom        = "hex"
    geom        = "quad"
    cond        = 2
    mu          = 0.700
    alpha       = 1.010
    beta        = alpha

    lattice     = grid(mu, N, N, alpha, beta, geom, cond)
    fig, ax1    = plt.subplots()

    mus         = np.arange(0.6, 1.001, 0.01)
    alphas      = np.arange(1.00, 1.0301, 0.0001)
    print(mus.size, alphas.size)
#    X       = mus
#    Y       = 1.02*np.ones(mus.size)
    X, Y        = np.meshgrid(mus, alphas)
    #variables   = np.stack((X.flatten(), Y.flatten() ), axis=-1 )

    #ax2         = ax1.twinx()
    #period_eval(lattice, ax1, ax2)
    phase_eval(lattice, ax1, X, Y)
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

def phase_eval(lattice, ax, X, Y):
    directory   = lattice.data_directory()
    latt_dir    = directory[0] + directory[1] + directory[2]
    var         = []
    variables   = np.stack((X, Y ), axis=2 )
    Z           = np.zeros(X.shape)

    directory       = lattice.data_directory()
    name            = directory[-2][:-1]
    directory[-2]   = 'phase_space_plots/'
    directory       = np.delete(directory, -1)
    directory       = ''.join(directory )
    data_ex     = os.path.exists(directory + name + '.npy')

    if data_ex:
        X, Y, Z = np.load(directory + name + '.npy')
    else:
        for idx in np.ndindex(X.shape ):

            set_of_var      = variables[idx]
            lattice.mu      = set_of_var[0]
            lattice.alpha   = set_of_var[1]
            lattice.beta    = lattice.alpha
            exists          = lattice.load()
            if exists:
                Z[idx]      = np.sqrt(lattice.var[-1][1] )/lattice.mu

    c   = ax.pcolormesh(X, Y, Z, shading='auto')
    clb = plt.colorbar(c, ax=ax)
    clb.ax.set_title('normed standard deviation')
    
    ax.set_xlabel('Average filling mu')
    ax.set_ylabel('Aggregation strength alpha')
    ax.grid(True)

    plt.savefig(directory + name + '.pdf')
    np.save(directory + name, [X, Y, Z] )


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
        os.makedirs(var_dir)
    
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
