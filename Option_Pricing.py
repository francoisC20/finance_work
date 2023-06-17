import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

class Option :
    
    def __init__(self,S0,K,r,sigma,T,is_call):
        """
        Builder of option type with following parameters:
        S0 : Initial Value of Underlying
        sigma : Volatility of underlying
        K : Strike price
        T : Time to maturity (expiration of the option)
        option_type : must be "call" or "put"
        """
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.r = r
        self.T = T
        self.is_call = is_call
    
        
    def plot_payoff(self,Smin,Smax,Ns):
        S = np.linspace(Smin,Smax,Ns+1)
        if self.is_call:
            payoff = np.maximum(S-self.K,0)
        else:
            payoff = np.maximum(self.K-S,0)
        plt.plot(S,payoff,color='r',label='Expiration payoff')
        
      

            
         
        
    def Monte_Carlo(self,N):
        """
        Monte Carlo simulation
        N : Number of simulations (at least 1 million for LLN to be applicable)
        """ 
        payoff_T = np.zeros(N)
        z = z = scs.norm.rvs(size=N,loc=0,scale=1)
        for i in range(N):
            ST = self.S0*np.exp((self.r-self.sigma**2/2)*self.T+self.sigma*np.sqrt(self.T)*z[i])
            if self.is_call:
                payoff_T[i] = max(ST-self.K,0)
            else:
                payoff_T[i] = max(self.K-ST,0)
        expected_payoff_T = np.mean(payoff_T)
        initial_value = np.exp(-self.r*self.T)*expected_payoff_T
        return initial_value
    
    def GMB_plot(self,N,parameter=""):
        """
        N : number of paths
        parameter : Component of the GMB process that we want to vizualize the effect
        """
        time, time_step = np.linspace(0,self.T,1000,retstep=True)
        gmb_path = np.zeros((1000,N))
        gmb_path[0,:] = self.S0*np.ones(N)
        if(parameter==""):
            for j in range(N):
                for i in range(len(time)-1):
                    gmb_path[i+1,j] = gmb_path[i,j]*np.exp((self.r-\
                                      0.5*self.sigma**2)*time_step+self.sigma*scs.norm.rvs(0,1,1)*np.sqrt(time_step))
                   
            plt.hist(gmb_path[-1,:],bins=50,density=True)
            plt.title('Underlying price at expiration t=T')    
                #plt.plot(time,gmb_path[:,j])
                #plt.xlabel('Time (years)')
                #plt.ylabel('Underlying price')
        elif(parameter=="mu"):
            color=['r','b','k']
            mu_range = np.linspace(0.05,0.15,3)
            for k in range(3):
                for j in range(N):
                    for i in range(len(time)-1):
                        gmb_path[i+1,j] = gmb_path[i,j]*np.exp((mu_range[k]-\
                                          0.5*self.sigma**2)*time_step+self.sigma*scs.norm.rvs(0,1,1)*np.sqrt(time_step))
                plt.plot(time,gmb_path[:,k:k+N//3],color=color[k])
        elif(parameter=="sigma"):
            color=['r','b','k','m','g']
            sigma_range = np.linspace(0.1,0.3,3)
            for k in range(3):
                for j in range(N):
                    for i in range(len(time)-1):
                        gmb_path[i+1,j] = gmb_path[i,j]*np.exp((self.r-\
                                          0.5*sigma_range[k]**2)*time_step+sigma_range[k]*scs.norm.rvs(0,1,1)*np.sqrt(time_step))
                plt.plot(time,gmb_path[:,k:k+N//3],color=color[k],label='sigma='+str(100*sigma_range[k])+'%')
                plt.legend()
        plt.xlabel('Time (years)')
        plt.ylabel('Underlying price')
    
    
    def BS_formula(self):
        d1 = ((self.r + 0.5 * self.sigma**2) * self.T - np.log(self.K / self.S0)) / (self.sigma * np.sqrt(self.T))
        d2 = ((self.r - 0.5 * self.sigma**2) * self.T - np.log(self.K / self.S0)) / (self.sigma * np.sqrt(self.T))
        if self.is_call:
            p = self.S0 * scs.norm.cdf(d1) - np.exp(-self.r * self.T) * self.K * scs.norm.cdf(d2) 
        else:               
            p = np.exp(-self.r * self.T) * self.K * scs.norm.cdf(-d2) - self.S0 * scs.norm.cdf(-d1)
        return p
 

        
            

