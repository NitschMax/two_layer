import numpy as np
import matplotlib.pyplot as plt
import data_dir
import os
from two_layer import grid

def main():
    N           = 100
    mu          = .7
    geom        = "hex"
    geom        = "quad"
    cond        = 0
    lattice     = grid(mu, N, N, geom, cond)

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
    
    plt.scatter(mu, var_upp_n, label='upper layer' )
    plt.scatter(mu, var_down_n, marker='x', label='lower layer')
    plt.grid(True)
    plt.xlabel("mu")
    plt.ylabel("normed variance")
    np.savetxt(var_dir + geom + '_variance_phases.txt', np.transpose([mu, var_down_n, var_upp_n] ) )
    plt.legend()

    plt.savefig(var_dir + geom + '_variance_phases.pdf')
    plt.show()
    

if __name__ == "__main__":
    main()
