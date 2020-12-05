import numpy as np
import matplotlib.pyplot as plt
import data_dir
import os

def main():
    N           = 100
    directory   = data_dir.get_dir()
    print(np.load(directory + str('Nk1-{}_Nk2-{}/x{:0.4f}/').format(N, N, 0.5) + "variance.npy" )[-4:] )
    directory   += str('Nk1-{}_Nk2-{}/').format(N, N)
    os.chdir(directory)
    files       = os.listdir(directory)

    var         = np.array([[float(crit_value[1:]), np.mean(np.load(crit_value + "/variance.npy" )[-100:])] for crit_value in files] )
    var         = var[var[:,0].argsort() ]
    plt.plot(var[:, 0], var[:, 1])
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("variance")
    plt.show()
    

if __name__ == "__main__":
    main()
