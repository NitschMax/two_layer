import data_dir
import os
from two_layer import grid
import matplotlib.pyplot as plt
import numpy as np

def main():
    N           = 100
    mu          = .7
    geom        = "hex"
    geom        = "quad"
    cond        = 0
    lattice     = grid(mu, N, N, geom, cond)
    fig, ax1    = plt.subplots()
    ax2         = ax1.twinx()

    #period_eval(lattice, ax1, ax2)
    var_eval(lattice, ax1)
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
    ax1.set_ylabel('Periodicity with max 1000')
    ax1.grid(True)
    plt.savefig('plots/periodicities.pdf')
    os.chdir(old_dir)


def var_eval(lattice, ax):
    directory   = lattice.data_directory()
    latt_dir    = directory[0] + directory[1] + directory[2]
    files       = os.listdir(latt_dir)
    var         = []

    for mu_direct in files:
        mu      = float(mu_direct[2:] )
        data    = np.array([np.load(latt_dir + mu_direct + "/variance_" + str(i) + ".npy" )[-1] for i in range(1,11)] )
        data    = np.array([np.concatenate( ([mu], np.load(latt_dir + mu_direct + "/variance_" + str(i) + ".npy" )[-1] )) for i in range(1,11)] )
        var.append(data)

    var         = np.array(var)
    var         = var.reshape(-1,3)
    var         = var[var[:,0].argsort() ]
    print(var.shape )
    print(var)
    mu          = var[:, 0]
    var_down    = var[:, 1]
    var_down_n  = var_down/mu**2
    var_upp     = var[:, 2]
    var_upp_n   = var_upp/mu**2

    var_dir     = "variances/" + directory[1] + directory[2]
    if not os.path.exists(var_dir):
        os.mkdir(var_dir)
    
    ax.scatter(mu, var_upp_n, marker='.', label='upper layer' )
    ax.scatter(mu, var_down_n, marker='.', label='lower layer')
    ax.grid(True)
    ax.set_xlabel("mu")
    ax.set_ylabel("normed variance")
    np.savetxt(var_dir + lattice.geom + '_variance_phases.txt', np.transpose([mu, var_down_n, var_upp_n] ) )
    ax.legend()

    plt.savefig(var_dir + lattice.geom + '_variance_phases.pdf')
    

if __name__ == "__main__":
    main()
