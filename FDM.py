import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

#--------------------- PARENT 1 -------------------------------#

class OptionsPricing(object):
    
    def __init__(self, S0, K, r, T, sigma, is_call=True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.is_call = is_call
        
#--------------------------------------------------------------#
#----------------------- SON LEVEL 1 --------------------------#

class FDM_Option(OptionsPricing):
    """
    Finite difference class
    """

    def __init__(self, S0, K, r, T, sigma, Smax, M, N, is_call=True):
        super(FDM_Option, self).__init__(S0, K, r, T, sigma, is_call)
        self.Smax = Smax
        self.M, self.N = int(M), int(N)  # Ensure M&N are integers

        self.dS = Smax / float(self.M)
        self.dt = T / float(self.N)
        self.iValues = np.arange(1, self.M)
        self.jValues = np.arange(self.N)
        self.grid = np.zeros(shape=(self.M+1, self.N+1)) # grid is M+1 by N+1
        self.SValues = np.linspace(0, Smax, self.M+1)

    def _fix_boundary_conditions_(self):
        pass

    def _coefficients_(self):
        pass

    def _fill_grid_(self):
        """  Iterate the grid backwards in time """
        pass

    def _interpolate_(self):
        """
        Use piecewise linear interpolation on the initial
        grid column to get the closest price at S0.
        """
        return np.interp(self.S0,
                         self.SValues,
                         self.grid[:, 0])

    def price(self):
         '''
        return the price of the option with corresponding given spot price 
        '''
        self._coefficients_()
        self._fix_boundary_conditions_()
        self._fill_grid_()
        return self._interpolate_()
    
    def grid_price(self):
        '''
        return the whole option price grid 
        '''
        self._coefficients_()
        self._fix_boundary_conditions_()
        self._fill_grid_()
        return self.grid
    
    def plot_price_surface(self):
        """
        Create 3D plot to see price surface of the option
        """
        self._coefficients_()
        self._fix_boundary_conditions_()
        self._fill_grid_()
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(121, projection='3d')
        t = self.dt * np.arange(self.N+1)
        X, Y = np.meshgrid(t,self.SValues)
        ax.plot_surface(Y, X, self.grid, cmap=cm.ocean)
        ax.set_title("BS Option price surface")
        ax.set_xlabel("Underlying Price"); ax.set_ylabel("time (years)"); ax.set_zlabel("V")
        if self.is_call:
            ax.view_init(20, 210) # this function rotates the 3d plot
        else:
            ax.view_init(20, -400)
            
        plt.show()
        
#--------------------------------------------------------------#
#----------------------- SON LEVEL 2 --------------------------#

class FDM_Explicit(FDM_Option):
    
    def _coefficients_(self):
        self.alpha = 0.5*self.dt * (self.sigma**2 * self.iValues**2 - self.r * self.iValues)
        self.beta  = - self.dt * (self.sigma**2 * self.iValues**2 + self.r)
        self.gamma = 0.5*self.dt * (self.sigma**2 * self.iValues**2 + self.r * self.iValues)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                      np.diag(1 + self.beta) + \
                      np.diag(self.gamma[:-1], 1)
        
    def _fix_boundary_conditions_(self):
        # terminal condition
        if self.is_call:
            self.grid[:, -1] = np.maximum(self.SValues - self.K, 0)
        else:
            self.grid[:, -1] = np.maximum(self.K - self.SValues, 0)
            
        # side boundary conditions
        self.coeffs[0,   0] += 2*self.alpha[0]
        self.coeffs[0,   1] -= self.alpha[0]
        self.coeffs[-1, -1] += 2*self.gamma[-1]
        self.coeffs[-1, -2] -= self.gamma[-1]
        
    def _fill_grid_(self):
        for j in reversed(self.jValues):
            self.grid[1:-1, j] = np.dot(self.coeffs, self.grid[1:-1, j+1])
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
            
#--------------------------------------------------------------#
#----------------------- SON LEVEL 3 --------------------------#
            
class FDM_Implicit(FDM_Explicit):
     '''
     we use the Lu method with metrix notations 
     '''
    
    def _coefficients_(self):
        self.alpha =  0.5*self.dt * (self.r * self.iValues - self.sigma**2 * self.iValues**2)
        self.beta  =  self.dt * (self.r + self.sigma**2 * self.iValues**2)
        self.gamma = -0.5*self.dt * (self.r * self.iValues + self.sigma**2 * self.iValues**2)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                      np.diag(1 + self.beta) + \
                      np.diag(self.gamma[:-1], 1)
        
    def _fill_grid_(self):           
        P, L, U = linalg.lu(self.coeffs)
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, self.grid[1:-1, j+1])
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]

#--------------------------------------------------------------#
#----------------------- SON LEVEL 3 --------------------------#

class FDM_Crank_Nicolson(FDM_Explicit):
    '''
    we use the Lu method with metrix notations 
    '''
    
    def _coefficients_(self):
        self.alpha = 0.25*self.dt * (self.sigma**2 * self.iValues**2 - self.r * self.iValues)
        self.beta  = -0.5*self.dt * (self.sigma**2 * self.iValues**2 + self.r)
        self.gamma = 0.25*self.dt * (self.sigma**2 * self.iValues**2 + self.r * self.iValues)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                       np.diag(1 + self.beta) + \
                       np.diag(self.gamma[:-1], 1)
        self.coeffs_ = np.diag(-self.alpha[1:], -1) + \
                       np.diag(1 - self.beta) + \
                       np.diag(-self.gamma[:-1], 1)

                       
    def _fix_boundary_conditions_(self):
        super(FDM_Crank_Nicolson, self)._fix_boundary_conditions_()
        self.coeffs_[0,   0] -= 2*self.alpha[0]
        self.coeffs_[0,   1] += self.alpha[0]
        self.coeffs_[-1, -1] -= 2*self.gamma[-1]
        self.coeffs_[-1, -2] += self.gamma[-1]

    def _fill_grid_(self):           
        P, L, U = linalg.lu(self.coeffs_)
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, np.dot(self.coeffs, self.grid[1:-1, j+1]))
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]