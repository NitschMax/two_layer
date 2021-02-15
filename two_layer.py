import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.sparse as sparse
import data_dir
from matplotlib.animation import FuncAnimation

class grid:
    ##### Let the lattice introduce itself
    def greet(self):
        print("Hello World, I am a grid of heigth {} and length {}! My average filling is {:1.4f} and I have a {} geometry. Alpha is {:1.3f} and beta {:1.3f}." \
                .format(self.h, self.l, self.mu, self.geom, self.alpha, self.beta) )

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

        return ['lattice_data/', 'condition_' + str(self.cond) + '/', self.geom + '_Nk1-{}_Nk2-{}/'.format(self.h,self.l), self.data_clarification() ]

    def data_clarification(self):
        return 'mu-{:0.4f}_alpha-{:1.4f}_beta-{:1.4f}/'.format(self.mu,self.alpha,self.beta )

    ##### Load an already calculated lattice with its lattest occupation and variancies
    def load(self):
        directory   = "".join(self.data_directory() )
        index       = 1

        if not os.path.exists(directory):
            print("There is no data available for this lattice configuration.")
            return False
        else:
            name        = "occupation_down_{:1.0f}.npy".format(index)
            self.down   = np.load(directory + name)
            name        = "occupation_upp_{:1.0f}.npy".format(index)
            self.upp    = np.load(directory + name)
            name        = "variance_{:1.0f}.npy".format(index)
            self.var    = list(np.load(directory + name) )
            self.time   = len(self.var )
            return True

    ##### Save the lattice and its variancies
    def save(self):
        directory   = "".join(self.data_directory() )

        if not os.path.exists(directory):
            os.makedirs(directory)
        files       = os.listdir(directory)
        files       = list(filter(var_in, files) )
        number      = int(len(files ) ) + 1
        
        name        = "occupation_down_{:1.0f}.npy".format(number)
        np.save(directory + name, self.down)
        name        = "occupation_upp_{:1.0f}.npy".format(number)
        np.save(directory + name, self.upp)
        name        = "variance_{:1.0f}.npy".format(number)
        np.save(directory + name, self.var)

        if self.period != -1:
            name    = "periodicity_{:1.0f}.npy".format(number)
            np.save(directory + name, self.period)


    ##### Plot the lattice and its variancies
    def plot(self, save_plot=False):
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))  = plt.subplots(2,3)
        c1                  = ax1.pcolor(self.upp, cmap='RdBu')
        c2                  = ax4.pcolor(self.down, cmap='RdBu', vmax=2.5*self.mu)
        normed_var          = np.array(self.var)/self.mu**2
        ax2.set_title('upper layer')
        ax2.set_xlabel('timestep')
        ax2.set_ylabel('standard deviation')

        ax5.set_title('lower layer')
        ax5.set_xlabel('timestep')
        ax5.set_ylabel('standard deviation')

        ax3.set_xlabel('moisture')
        ax3.set_ylabel('number of cells')

        ax6.set_xlabel('moisture')
        ax6.set_ylabel('number of cells')

        ax2.plot(normed_var[:,1], )
        ax5.plot(normed_var[:,0], )
        ax3.hist(self.upp.flatten(), bins=20)
        ax6.hist(self.down.flatten(), bins=20)
        fig.colorbar(c1, ax=ax1)
        fig.colorbar(c2, ax=ax4)
        fig.tight_layout()

        if save_plot:
            sub_dirs        = self.data_directory()
            directory       = "".join(sub_dirs )
            name            = sub_dirs[-1]
            name            = name[:-1] + '.pdf'
            print(name)

            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory + name)

        plt.show()

    ##### Routine to build an animation out of the simulation of a grid
    def animation(self, n, k=1, show_ani=True, save_ani=False):
        fig, (ax1,ax2)     = plt.subplots(1, 2, figsize=[12.8, 9.6])
        x       = list(range(self.l+1 ) )
        y       = x
        mesh1   = ax1.pcolormesh(x, y, self.upp, cmap='RdBu', vmin=0)
        mesh2   = ax2.pcolormesh(x, y, self.down, cmap='RdBu', vmin=0)
        ax1.set_title('upper layer')
        ax2.set_title('lower layer')
        fig.colorbar(mesh1, ax=ax1)
        fig.colorbar(mesh2, ax=ax2)


        def animate(i):
            self.run(k)

            mesh1.set_array(self.upp.flatten() )
            mesh2.set_array(self.down.flatten() )

            #### furhter possibilities which are not yet included
            #f_d.set_color(colors(i))
            #temp.set_text(str(int(T[i])) + ' K')
            #temp.set_color(colors(i))

        ani = FuncAnimation(fig=fig, func=animate, frames=n, interval=40, repeat=True)
        if save_ani:
            directory       = self.data_directory()
            spec_name       = directory[-1]
            spec_name       = spec_name[:-1]
            print(spec_name)
            directory[0]    = 'animations/'
            directory       = "".join(directory )

            if not os.path.exists(directory):
                os.makedirs(directory)

            print('Save animation in', directory)
            ani.save(directory + spec_name + '_animation.mp4', writer='ffmpeg')

        if show_ani:
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
            wert    = self.time_step_ind()

            if wert == 0:
                print('Simulation stopped because of convergence after {} steps'.format(self.time) )
                return

    def latt_var_period(self, n, k):
        self.greet()
        
        self.fill_random()
        self.run(n)

        self.save()

        initial_pattern = self.topple()
        for h in range(1, k+1):
            self.time_step_ind()
            candidate  = self.topple()
            if np.array_equal(initial_pattern, candidate):
                self.period     = h
                break
            if h == k:
                self.period     = False


    def periodicity(self, n, k):
        result  = []
        for i in range(10):
            print(i)
            self.fill_random()
            self.run(n)
            initial_pattern = self.topple()
            for h in range(1, k+1):
                self.time_step_ind()
                candidates  = self.topple()
                if np.array_equal(initial_pattern, candidates):
                    result.append([self.mu, h] )
                    break
                if h == k:
                    result.append([self.mu, False])

        directory   = self.data_directory()
        directory   = "periodicities/" + directory[1] + directory[2]

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        name        = "periodicities_mu-{:1.4f}".format(self.mu)
        np.save(directory + name, result )
            
    def topple(self):
        if self.cond == 0:
            candidates  = np.array(np.where(self.down+self.upp > 1) )
        elif self.cond == 1:
            candidates  = np.array(np.where(self.down*self.upp > 1) )
        elif self.cond == 2:
            candidates  = np.array(np.where(self.down > 1) )

        return candidates

    ##### Last version of the time_step algorithm, purely array indexing, by way the fastest version
    def time_step_ind(self):
        candidates  = self.topple()
        if candidates.size == 0:
            return 0                                                    # Break the routine of there is no spilling possible

        fillings    = self.down[candidates[0], candidates[1]]
        self.down[candidates[0], candidates[1]]  = 0
        if self.geom == 'quad':
            steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1] ] )
        elif self.geom == 'hex':
            steps       = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, 1] ] )
			
        candidates  = np.transpose(candidates)
        neighboors  = np.mod(candidates.reshape((candidates.shape[0], 1, candidates.shape[1]) ) + steps, self.l )
        neighboors  = np.transpose(neighboors, (1, 2, 0))
        num_neigh   = len(steps)+1
        for el in neighboors:
            self.down[el[0], el[1]]             += self.alpha*fillings/num_neigh             # This is necessary to get overspillings in the same cell right

        candidates  = np.transpose(candidates)
        self.upp[candidates[0], candidates[1]]  += self.beta*fillings/num_neigh

        total_one   = self.a*self.mu

        total_upp   = np.sum(self.upp)
        exc_upp     = total_upp-total_one
        percentage  = exc_upp/total_upp
        self.down   += percentage*self.upp
        self.upp    -= percentage*self.upp

        total_down  = np.sum(self.down)
        total_exc   = total_down-total_one
        percentage  = total_exc/total_down
        self.down   -= percentage*self.down

        self.var.append([np.var(self.down), np.var(self.upp) ] )
        self.time   += 1
        return 1

    def __init__(self, mu, heigth, length, alpha, beta, geom="quad", cond=0):
        self.h      = heigth                        #heigth of the grid
        self.l      = length                        #length of the grid
        self.a      = self.h*self.l                 #area of the grid
        self.mu     = mu                            #average filling of the grid
        self.time   = 0                             #current time step of the grid
        self.var    = []                            #variance of the grid
        self.alpha  = alpha                         #toppling excession in horizontal direction
        self.beta   = beta                          #toppling excession in vertical direction
        self.period = -1

        if geom in ["quad", "hex"]:
            self.geom   = geom                          #Geometry of the lattice
        else:
            print('Choosen geometry does not exist. Quadratic geometric choosen as default')
            self.geom   = "quad"

        if cond in [0, 1, 2]:
    	    self.cond   = cond
        else:
            print('Choosen condition does not exist. Plus condition is choosen as default')
            self.cond   = 0
					
        self.upp    = np.zeros((self.h, self.l) )   #occupations, to get non-zero ocupations use the fill_* fuctions
        self.down   = np.zeros((self.h, self.l) )   #occupations, to get non-zero ocupations use the fill_* fuctions

    
def var_in(word):
    return 'var' in word



