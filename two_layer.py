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
        self.down      = np.random.rand(self.h, self.l)
        self.down      *= self.mu/self.down.mean()
        self.upp       = np.random.rand(self.h, self.l)
        self.upp       *= self.mu/self.upp.mean()
        self.time   = 0
        self.var    = [[np.var(self.down), np.var(self.upp)] ]

    def fill_checkerboard(self):
        self.down      = np.indices((self.h, self.l) ).sum(axis=0) % 2 
        self.down      *= self.mu/self.down.mean()
        self.upp        = np.indices((self.h, self.l) ).sum(axis=0) % 2 
        self.upp        *= self.mu/self.upp.mean()
        self.time   = 0
        self.var    = [[np.var(self.down), np.var(self.upp)] ]

    ###### Modify a lattice by hand
    def modify(self, i, j, value):
        if (i >= self.h) or (i < 0) or (j >= self.l) or (j < 0):
            print("This modification is not possible.")
        if not (isinstance(i, int) and isinstance(j, int) ):
            print("This modification is not possible.")
        else:
            self.down[i, j]    = value

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
            self.down      = np.load(directory + name)
            name        = "variance.npy"
            self.var    = list(np.load(directory + name) )
            self.time   = len(self.var )

    ##### Save the lattice and its variancies
    def save(self):
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        name        = "occupation.npy"
        np.save(directory + name, self.down)
        name        = "variance.npy"
        np.save(directory + name, self.var)
    
    ##### Plot the lattice and its variancies
    def plot(self):
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))  = plt.subplots(2,3)
        c2                  = ax1.pcolor(self.upp, cmap='RdBu', vmin=0)
        c1                  = ax4.pcolor(self.down, cmap='RdBu', vmin=0)
        ax2.plot(self.var, )
        ax2.legend(['down', 'upp'])
        ax3.hist(self.upp.flatten(), bins=20)
        ax6.hist(self.down.flatten(), bins=20)
        fig.colorbar(c1, ax=ax1)
        fig.colorbar(c2, ax=ax4)
        fig.tight_layout()
        plt.show()

    ##### Routine to build an animation out of the simulation of a grid
    def animation(self, n):
        fig, (ax1,ax2)     = plt.subplots(1, 2)
        x       = list(range(self.l+1 ) )
        y       = x
        mesh1   = ax1.pcolormesh(x, y, self.upp, cmap='RdBu', vmin=0)
        mesh2   = ax2.pcolormesh(x, y, self.down, cmap='RdBu', vmin=0)


        def animate(i):
            if self.geom == "quad":
                self.time_step_ind()
            elif self.geom == "hex":
                self.time_step_hex()
            else:
                print("Something went running with the geometry of the lattice")

            mesh1.set_array(self.upp.flatten() )
            mesh2.set_array(self.down.flatten() )

            #### furhter possibilities which are not yet included
            #f_d.set_color(colors(i))
            #temp.set_text(str(int(T[i])) + ' K')
            #temp.set_color(colors(i))

        ani = FuncAnimation(fig=fig, func=animate, frames=n, interval=100, repeat=True)
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            os.makedirs(directory)

        ani.save(directory + 'animation.mp4')
        plt.show()

    ##### A routine to check for a simple test of one time_step
    def test_run(self):
        middle      = int(self.l/2)
        self.down      = np.zeros((self.h, self.l) )

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
        candidates  = np.array(np.where(self.down > 1) )
        if candidates.size == 0:
            return 0                                                    # Break the routine of there is no spilling possible

        fillings    = self.down[candidates[0], candidates[1]]
        self.down[candidates[0], candidates[1]]  = 0
        steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1] ] )
        candidates  = np.transpose(candidates)
        neighboors  = np.mod(candidates.reshape((candidates.shape[0], 1, candidates.shape[1]) ) + steps, self.l )
        neighboors  = np.transpose(neighboors, (1, 2, 0))
        for el in neighboors:
            self.down[el[0], el[1]]             += fillings/(len(steps)+1)             # This is necessary to get overspillings in the same cell right

        candidates  = np.transpose(candidates)
        self.upp[candidates[0], candidates[1]]  += fillings/(len(steps)+1)

        total_dist  = np.sum(fillings )/(len(steps)+1)
        total_upp   = self.a*self.mu+total_dist
        percentage  = total_dist/total_upp
        down_dist   = self.upp*percentage
        self.down   += down_dist
        self.upp    *= (1-percentage)

        self.var.append([np.var(self.down), np.var(self.upp) ] )
        self.time   += 1
        return 1

    ##### time step with hexagon symmetry
    def time_step_hex(self):
        candidates  = np.array(np.where(self.down > 1) )
        if candidates.size == 0:
            return 0                                                    # Break the routine of there is no spilling possible

        fillings    = self.down[candidates[0], candidates[1]]
        self.down[candidates[0], candidates[1]]  = 0
        steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1] ] )
        candidates  = np.transpose(candidates)
        neighboors  = np.mod(candidates.reshape((candidates.shape[0], 1, candidates.shape[1]) ) + steps, self.l )
        neighboors  = np.transpose(neighboors, (1, 2, 0))
        for el in neighboors:
            self.down[el[0], el[1]]             += fillings/(len(steps)+1)             # This is necessary to get overspillings in the same cell right

        candidates  = np.transpose(candidates)
        self.upp[candidates[0], candidates[1]]  += fillings/(len(steps)+1)

        total_dist  = np.sum(fillings )/(len(steps)+1)
        total_upp   = self.a*self.mu+total_dist
        percentage  = total_dist/total_upp
        down_dist   = self.upp*percentage
        self.down   += down_dist
        self.upp    *= (1-percentage)

        self.var.append([np.var(self.down), np.var(self.upp) ] )
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
        self.upp    = np.zeros((self.h, self.l) )   #occupations, to get non-zero ocupations use the fill_* fuctions
        self.down   = np.zeros((self.h, self.l) )   #occupations, to get non-zero ocupations use the fill_* fuctions

    











