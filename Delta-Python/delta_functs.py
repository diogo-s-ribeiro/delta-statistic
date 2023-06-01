
import os, math
import numpy as np
from scipy.stats import entropy
from numba import njit, float64, int64

############ Delta Functions [1] ############
# Metropolis-Hastings step for alpha parameter
@njit(float64(float64, float64, float64[::1], float64, float64))
def mhalpha(a,b,x,l0,se):
    '''a = The current value of the alpha parameter.
    b    = The current value of the beta parameter.
    x    = An array of data points used in the acceptance ratio computations, after uncertainty is calculated.
    l0   = A constant value used in the acceptance ratio computations.
    se   = The standard deviation used for the random walk in the Metropolis-Hastings algorithm.'''

    a1   = np.exp(np.random.normal(np.log(a),se, 1))[0]
    lp_a = np.exp( (len(x)*(math.lgamma(a1+b)-math.lgamma(a1)) - a1*(l0-np.sum(np.log(x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(a)) - a*(l0-np.sum(np.log(x)))) )
    r    = min( 1, lp_a ) 

    # Repeat until a valid value is obtained
    while (np.isnan(lp_a) == True):
        a1   = np.exp(np.random.normal(np.log(a),se, 1))[0]
        lp_a = np.exp( (len(x)*(math.lgamma(a1+b)-math.lgamma(a1)) - a1*(l0-np.sum(np.log(x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(a)) - a*(l0-np.sum(np.log(x)))) )
        r    = min( 1, lp_a )
    
    # Accept or reject based on the acceptance ratio
    if np.random.uniform(0,1) < r:
        return a1
    else:
        return a


# Metropolis-Hastings step for beta parameter
@njit(float64(float64, float64, float64[::1], float64, float64))
def mhbeta(a,b,x,l0,se):
    '''a = The current value of the alpha parameter.
    b    = The current value of the beta parameter.
    x    = An array of data points used in the acceptance ratio computations, after uncertainty is calculated.
    l0   = A constant value used in the acceptance ratio computations.
    se   = The standard deviation used for the random walk in the Metropolis-Hastings algorithm.'''
    
    b1   = np.exp(np.random.normal(np.log(b),se,1))[0]
    lp_b = np.exp( (len(x)*(math.lgamma(a+b1)-math.lgamma(b1)) - b1*(l0-np.sum(np.log(1-x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(b)) - b*(l0-np.sum(np.log(1-x)))) )
    r    = min( 1, lp_b )
    
    # Repeat until a valid value is obtained
    while (np.isnan(lp_b) == True):
        b1   = np.exp(np.random.normal(np.log(b),se,1))[0]
        lp_b = np.exp( (len(x)*(math.lgamma(a+b1)-math.lgamma(b1)) - b1*(l0-np.sum(np.log(1-x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(b)) - b*(l0-np.sum(np.log(1-x)))) )
        r    = min( 1, lp_b )
    
    # Accept or reject based on the acceptance ratio
    if np.random.uniform(0,1) < r:
        return b1
    else:
        return b


# Metropolis-Hastings algorithm using alpha and beta
@njit(float64[:, ::1](float64, float64, float64[::1], float64, float64, int64, int64, int64))
def emcmc(alpha,beta,x,l0,se,sim,thin,burn):
    '''alpha = The initial value of the alpha parameter.
    beta     = The initial value of the beta parameter.
    x        = An array of data points used in the acceptance ratio computations, after uncertainty is calculated.
    l0       = A constant value used in the acceptance ratio computations.
    se       = The standard deviation used for the random walk in the Metropolis-Hastings algorithm.
    sim      = The number of total iterations in the Markov Chain Monte Carlo (MCMC) simulation.
    thin     = The thinning parameter, i.e., the number of iterations to discard between saved samples.
    burn     = The number of burn-in iterations to discard at the beginning of the simulation.'''

    n_size = np.linspace(burn, sim, int((sim - burn) / thin + 1))
    usim   = np.round(n_size, 0, np.empty_like(n_size))
    gibbs  = []
    p      = 0

    for i in range(sim+1):
        alpha = mhalpha(alpha,beta,x,l0,se)
        beta  = mhbeta(alpha,beta,x,l0,se)
        
        if i == usim[p]:
            gibbs.append((alpha, beta))
            p += 1
            
    gibbs = np.asarray(gibbs)      
    return gibbs


# Calculate uncertainty using different types
def entropy_type(prob, ent_type):
    '''prob  = A matrix of ancestral probabilities.
    ent_type = A string indicating the type of entropy calculation. (options: 'LSE', 'SE', or any other value for Gini impurity).'''
    
    # Linear Shannon Entropy
    if ent_type == 'LSE':
        k    = np.shape(prob)[1]
        prob = np.asarray(np.where(prob<=(1/k), prob, prob/(1-k) - 1/(1-k)))
        tent = np.sum(prob, 1)
        
        # Ensure absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))
        
        return tent

    # Shannon Entropy
    elif ent_type == 'SE':
        k    = np.shape(prob)[1]
        tent = entropy(prob, base=k, axis=1)
        
        # Ensure absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))

        return tent

    # Ginni Impurity
    else:
        k    = np.shape(prob)[1]
        tent = ((1 - np.sum(prob**2, axis=1))*k)/ (k - 1)
        
        # Ensure absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))

        return tent


# Calculate delta-statistic after an MCMC step
def delta(x,lambda0,se,sim,thin,burn,ent_type):
    '''x     = A matrix of ancestral probabilities.
    lambda0  = A constant value used in the acceptance ratio computations.
    se       = The standard deviation used for the random walk in the Metropolis-Hastings algorithm.
    sim      = The number of total iterations in the Markov Chain Monte Carlo (MCMC) simulation.
    thin     = The thinning parameter, i.e., the number of iterations to discard between saved samples.
    burn     = The number of burn-in iterations to discard at the beginning of the simulation.
    ent_type = A string specifying the type of entropy calculation (options: 'LSE', 'SE', or any other value for Gini impurity).'''
    
    mc1    = emcmc(np.random.exponential(),np.random.exponential(),entropy_type(x, ent_type),lambda0,se,sim,thin,burn)
    mc2    = emcmc(np.random.exponential(),np.random.exponential(),entropy_type(x, ent_type),lambda0,se,sim,thin,burn)
    mchain = np.concatenate((mc1,mc2), axis=0)
    
    deltaA = (np.mean(mchain[:,1]))/(np.mean(mchain[:,0]))
    
    return deltaA


''' ADDITIONAL INFORMATION

[1] Borges, R. et al. (2019). Measuring phylogenetic signal between categorical traits and phylogenies. Bioinformatics, 35, 1862-1869.

'''