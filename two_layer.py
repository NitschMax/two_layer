import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.sparse as sparse
import data_dir
from matplotlib.animation import FuncAnimation

class grid:
    ##### Let the lattice introduce itself
    def greet(self):
        print("Hello World, I am a grid of heigth {} and length {}! My average filling is {:1.4f} and I have a {} geometry.".format(self.h, self.l, self.mu, self.geom)  )

    ##### Several interesting starting configurations
    def fill_random(self):
        self.o      = 4*np.random.rand(self.h, self.l)
        self.o      *= self.mu/self.o.mean()
        self.time   = 0
        self.var    = [np.var(self.o)]

    def fill_checkerboard(self):
        self.o      = np.indices((self.h, self.l) ).sum(axis=0) % 2 
        self.o      *= self.mu/self.o.mean()
        self.time   = 0
        self.var    = [np.var(self.o)]

    def fill_random_checker(self):
        self.o      = (np.indices((self.h, self.l) ).sum(axis=0) % 2)*(1-.1*np.random.rand(self.h, self.l) )
        self.o      *= self.mu/self.o.mean()
        self.time   = 0
        self.var    = [np.var(self.o)]

    ###### Modify a lattice by hand
    def modify(self, i, j, value):
        if (i >= self.h) or (i < 0) or (j >= self.l) or (j < 0):
            print("This modification is not possible.")
        if not (isinstance(i, int) and isinstance(j, int) ):
            print("This modification is not possible.")
        else:
            self.o[i, j]    = value

    ##### This routine determine where the data is found on the machine
    def data_directory(self):
        directory   = data_dir.get_dir()

        if not os.path.exists(directory):
            os.mkdir(directory)
        os.chdir(directory)

        return ['lattice_data/', self.geom + '_Nk1-{}_Nk2-{}/'.format(self.h,self.l), 'mu{:0.4f}/'.format(self.mu) ]

    ##### Load an already calculated lattice with its lattest occupation and variancies
    def load(self):
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            print("There is no data available for this lattice configuration.")
        else:
            name        = "occupation.npy"
            self.o      = np.load(directory + name)
            name        = "variance.npy"
            self.var    = list(np.load(directory + name) )
            self.time   = len(self.var )

    ##### Save the lattice and its variancies
    def save(self):
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            os.makedirs(directory)

        name        = "occupation.npy"
        np.save(directory + name, self.o)
        name        = "variance.npy"
        np.save(directory + name, self.var)
    
    ##### Plot the lattice and its variancies
    def plot(self):
        fig, (ax1,ax2,ax3)  = plt.subplots(1,3)
        c                   = ax1.pcolor(self.o, cmap='RdBu', vmin=0)
        ax2.plot(self.var)
        ax3.hist(self.o.flatten(), bins=20)
        fig.colorbar(c, ax=ax1)
        fig.tight_layout()
        plt.show()

    ##### Routine to build an animation out of the simulation of a grid
    def animation(self, n):
        fig     = plt.figure(figsize=(6, 4))
        ax      = fig.add_subplot(111)
        x       = list(range(self.l+1 ) )
        y       = x
        f_d     = ax.pcolormesh(x, y, self.o, cmap='RdBu', vmin=0)

        def animate(i):
            self.time_step_ind()
            f_d.set_array(self.o.flatten() )

            #### furhter possibilities which are not yet included
            #f_d.set_color(colors(i))
            #temp.set_text(str(int(T[i])) + ' K')
            #temp.set_color(colors(i))

        ani = FuncAnimation(fig=fig, func=animate, frames=n, interval=100, repeat=True)
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            os.makedirs(directory)

        ani.save(directory + 'animation.mp4')
        #plt.show()

    ##### A routine to check for a simple test of one time_step
    def test_run(self):
        middle      = int(self.l/2)
        self.o      = np.zeros((self.h, self.l) )

        self.modify(self.l-1, self.l-1, 2)
        self.modify(0, 0, 2)
        self.modify(middle, middle, 2)
        self.modify(middle-1, middle-1, 2)

        self.plot()
        self.time_step_ind()
        self.plot()

    ##### A running routine, if the time_step algorithm returns 0, then the lattice has no possible spillings any more
    def run(self, N):
        for k in range(N):
            if self.geom == "quad":
                wert    = self.time_step_ind()
            elif self.geom == "hex":
                wert    = self.time_step_hex()
            else:
                print("Something went running with the geometry of the lattice")
                break


            if wert == 0:
                return

    ##### Last version of the time_step algorithm, purely array indexing, by way the fastest version
    def time_step_ind(self):
        candidates  = np.array(np.where(self.o > 1) )
        if candidates.size == 0:
            return 0                                                    # Break the routine of there is no spilling possible

        fillings    = self.o[candidates[0], candidates[1]]
        self.o[candidates[0], candidates[1]]  = 0
        steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1] ] )
        candidates  = np.transpose(candidates)
        neighboors  = np.mod(candidates.reshape((candidates.shape[0], 1, candidates.shape[1]) ) + steps, self.l )
        neighboors  = np.transpose(neighboors, (1, 2, 0))
        for el in neighboors:
            self.o[el[0], el[1]]    += fillings/len(steps)              # This is necessary to get overspillings in the same cell right
        self.var.append(np.var(self.o) )
        self.time   += 1
        return 1

    ##### time step with hexagon symmetry
    def time_step_hex(self):
        candidates  = np.array(np.where(self.o > 1) )
        if candidates.size == 0:
            return 0                                                    # Break the routine of there is no spilling possible

        fillings    = self.o[candidates[0], candidates[1]]
        self.o[candidates[0], candidates[1]]  = 0
        steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1] ] )
        candidates  = np.transpose(candidates)
        neighboors  = np.mod(candidates.reshape((candidates.shape[0], 1, candidates.shape[1]) ) + steps, self.l )
        neighboors  = np.transpose(neighboors, (1, 2, 0))
        for el in neighboors:
            self.o[el[0], el[1]]    += fillings/len(steps)                      # This is necessary to get overspillings in the same cell right
        self.var.append(np.var(self.o) )
        self.time   += 1
        return 1

    ##### First version of the time step algorithm, mixture of for loop and matrix multiplication
    def time_step(self):
        overspill   = np.transpose(np.where(self.o > 1) )
        update      = np.identity(self.h*self.l)
        vector      = self.o.flatten()

        for example in overspill:
            j           = example[0]
            l           = example[1]
            neighbors   = [j*self.l+np.mod(l-1, self.l), j*self.l+np.mod(l+1, self.l), np.mod(j-1, self.h)*self.l+l, np.mod(j+1, self.h)*self.l+l ]
            index                           = j*self.l+l
            spill                           = np.zeros((self.h*self.l, self.h*self.l) )
            spill[index, index]             = -1
            spill[neighbors[0], index]      = 1/4
            spill[neighbors[1], index]      = 1/4
            spill[neighbors[2], index]      = 1/4
            spill[neighbors[3], index]      = 1/4
            update                          += spill

        vector                          = np.matmul(update, vector)
        self.o                          = vector.reshape(self.h, self.l)

    ##### Second version of the time step algorithm, indexing with a matrix multiplication
    def time_step_mat(self):
        candidates  = np.transpose(np.where(self.o > 1) )
        if candidates.size == 0:
            return 0
        vector      = self.o.flatten()

        steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1] ] )
        vec_index   = np.transpose(np.array([self.l, 1]) )

        indices     = np.matmul(candidates, vec_index)
        neighbors   = np.array([np.matmul(np.mod(el + steps, self.l), vec_index) for el in  candidates] ) #.reshape((-1, 2) 
        indices     = indices.reshape((indices.size, 1))

        spill        = sparse.eye(self.a, format="csr").toarray()
        spill[neighbors, indices]   = 1/4
        spill[indices, indices]     = 0

        vector      = spill.dot(vector)
        self.o      = vector.reshape(self.h, self.l)
        self.var.append(np.var(self.o) )
        self.time   += 1
        return 1

    def __init__(self, mu, heigth, length, geom="quad"):
        self.h      = heigth                        #heigth of the grid
        self.l      = length                        #length of the grid
        self.a      = self.h*self.l                 #area of the grid
        self.mu     = mu                            #average filling of the grid
        self.time   = 0                             #current time step of the grid
        self.var    = []                            #variance of the grid
        if geom in ["quad", "hex"]:
            self.geom   = geom                          #Geometry of the lattice
        else:
            print('Choosen geometry does not exist. Quadratic geometric choosen as defaultl')
            self.geom   = "quad"
        self.o      = np.zeros((self.h, self.l) )   #occupations, to get non-zero ocupations use the fill_* fuctions

    











