import numpy as np
import matplotlib.pyplot as plt
import data_dir
import os
from sandpile import grid

def main():
    N           = 100
    mu          = .7
    geom        = "quad"
    geom        = "hex"
    lattice     = grid(mu, N, N, geom)
    directory   = lattice.data_directory()
    latt_dir    = directory[0] + directory[1]
    files       = os.listdir(latt_dir)

    var         = np.array([[float(mu[2:]), np.mean(np.load(latt_dir + mu + "/variance.npy" )[-4:])] for mu in files] )
    var         = var[var[:,0].argsort() ]
    print(var)
    mu          = var[1:, 0]
    variance    = var[1:, 1]
    normedVar   = variance/mu**2
    var_dir     = "variances/" + directory[1]
    if not os.path.exists(var_dir):
        os.mkdir(var_dir)

    plt.plot(mu, normedVar)
    plt.grid(True)
    plt.xlabel("mu")
    plt.ylabel("normed variance")
    np.savetxt(var_dir + geom + '_variance_phases.txt', np.transpose([mu, normedVar] ) )
    plt.savefig(var_dir + geom + '_variance_phases.pdf')
    plt.show()
    

if __name__ == "__main__":
    main()
