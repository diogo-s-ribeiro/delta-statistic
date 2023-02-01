
import os, math
import numpy as np
from scipy.stats import entropy
from numba import njit, float64, int64

############ Delta Functions [1] ############
@njit(float64(float64, float64, float64[::1], float64, float64))
def mhalpha(a,b,x,l0,se):
    a1   = np.exp(np.random.normal(np.log(a),se, 1))[0]
    lp_a = np.exp( (len(x)*(math.lgamma(a1+b)-math.lgamma(a1)) - a1*(l0-np.sum(np.log(x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(a)) - a*(l0-np.sum(np.log(x)))) )
    r    = min( 1, lp_a )

    while (np.isnan(lp_a) == True):
        a1   = np.exp(np.random.normal(np.log(a),se, 1))[0]
        lp_a = np.exp( (len(x)*(math.lgamma(a1+b)-math.lgamma(a1)) - a1*(l0-np.sum(np.log(x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(a)) - a*(l0-np.sum(np.log(x)))) )
        r    = min( 1, lp_a )
        
    if np.random.uniform(0,1) < r:
        return a1
    else:
        return a

@njit(float64(float64, float64, float64[::1], float64, float64))
def mhbeta(a,b,x,l0,se):
    b1   = np.exp(np.random.normal(np.log(b),se,1))[0]
    lp_b = np.exp( (len(x)*(math.lgamma(a+b1)-math.lgamma(b1)) - b1*(l0-np.sum(np.log(1-x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(b)) - b*(l0-np.sum(np.log(1-x)))) )
    r    = min( 1, lp_b )
    
    while (np.isnan(lp_b) == True):
        b1   = np.exp(np.random.normal(np.log(b),se,1))[0]
        lp_b = np.exp( (len(x)*(math.lgamma(a+b1)-math.lgamma(b1)) - b1*(l0-np.sum(np.log(1-x)))) - (len(x)*(math.lgamma(a+b)-math.lgamma(b)) - b*(l0-np.sum(np.log(1-x)))) )
        r    = min( 1, lp_b )
        
    if np.random.uniform(0,1) < r:
        return b1
    else:
        return b

@njit(float64[:, ::1](float64, float64, float64[::1], float64, float64, int64, int64, int64))
def emcmc(alpha,beta,x,l0,se,sim,thin,burn):
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


def entropy_type(prob, ent_type):
    if ent_type == 'LSE':
        k    = np.shape(prob)[1]
        prob = np.asarray(np.where(prob<=(1/k), prob, prob/(1-k) - 1/(1-k)))
        tent = np.sum(prob, 1)
        
        #absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))
        
        return tent

    elif ent_type == 'SE':
        k    = np.shape(prob)[1]
        tent = entropy(prob, base=k, axis=1)
        
        #absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))

        return tent

    else:
        k    = np.shape(prob)[1]
        tent = ((1 - np.sum(prob**2, axis=1))*k)/ (k - 1)
        
        #absolutes
        tent = np.asarray(np.where(tent != 0, tent, tent + np.random.uniform(0,1,1)/10000))
        tent = np.asarray(np.where(tent != 1, tent, tent - np.random.uniform(0,1,1)/10000))

        return tent

def delta(x,lambda0,se,sim,thin,burn,ent_type):
    mc1    = emcmc(np.random.exponential(),np.random.exponential(),entropy_type(x, ent_type),lambda0,se,sim,thin,burn)
    mc2    = emcmc(np.random.exponential(),np.random.exponential(),entropy_type(x, ent_type),lambda0,se,sim,thin,burn)
    mchain = np.concatenate((mc1,mc2), axis=0)
    
    deltaA = (np.mean(mchain[:,1]))/(np.mean(mchain[:,0]))
    
    return deltaA



############ Ancestral Probabilities ############
############ Input directly from external source (p.e. After running a Bayesian approach)
def read_file_anc(file_path, separator, rmv_col_name, rmv_row_name):
    array = np.genfromtxt(file_path, delimiter = separator)
    if rmv_col_name == True:
        array = np.delete(array, 0, axis=0)     # Remove Trait Name
    if rmv_row_name == True:
        array = np.delete(array, 0, axis=1)     # Remove Entity Name
    return array

############ Calculate using PastML [2]
from pastml.tree import read_tree
from pastml.acr import _validate_input, acr

def marginal(tree, data=None, data_sep='\t', id_index=0,
            columns=None, prediction_method='MPPA', model='F81',
            name_column=None, forced_joint=False, threads=0):

    if threads < 1:
        threads = max(os.cpu_count(), 1)

    roots, columns, column2states, name_column, age_label, parameters, rates = \
        _validate_input(tree, columns, name_column, data, data_sep, id_index)

    acr_results = acr(forest=roots, columns=columns, column2states=column2states,
                    prediction_method=prediction_method, model=model, column2parameters=parameters,
                    column2rates=rates, force_joint=forced_joint, threads=threads)
    
    leaf_names = read_tree(tree).get_leaf_names()
    marginal   = np.asarray( acr_results[0]['marginal_probabilities'].drop(leaf_names) )

    return marginal

############ Calculate using other software packages (p.e. ape package [3] in python through rpy2 [4])
# Multiple software packages are currently available that can perform ancestral state reconstruction. Choose the best suited for your needs! [5]


############ Example Variables ############
############ Delta
lambda0  = 0.1                                          # rate parameter of the proposal
se       = 0.5                                          # standard deviation of the proposal
sim      = 10000                                        # number of iterations
thin     = 10                                           # Keep only each xth iterate 
burn     = 100                                          # Burned-in iterates

############ Uncertainty method 
ent_type    = 'LSE'                                     # Linear Shannon Entropy [1]
# ent_type  = 'SE'                                      # Shannon Entropy [6] (normalized)
# ent_type  = 'GI'                                      # Gini Impurity [7] (normalized)

############ Ancestral Probabilities
path_input  = r".\input\FILE"                           # Path to input files      

## Input directly ##
file         = "test_anc.csv"                           # File with Ancestral Probabilities
file         = path_input.replace('FILE', file)         # Path + file
separator    = ","                                      # If File is (.csv); separator = ','
rmv_col_name = True                                     # Remove Trait Name
rmv_row_name = True                                     # Remove Entity Name

## PastML ##
data     = "States.txt"                                 # File containing tip/node annotations, in csv or tab format
data     = path_input.replace('FILE', data)             # Path + data
data_sep = ","                                          # If data is (.csv); separator = ','
tree     = "Trees.txt"                                  # File of the phylogenetic tree in a newick format
tree     = path_input.replace('FILE', tree)             # Path + tree
columns  = "State"                                      # Column to reconstruct ancestral states
method   = "MPPA"                                       # MPPA, MAP
model    = "F81"                                        # F81, JC, EFT


############ Calculate ############
# ap    = read_file_anc(file_path=file, separator=separator, rmv_col_name=rmv_col_name, rmv_row_name=rmv_row_name)
# ap    = marginal(data=data, data_sep=data_sep, columns=columns, tree=tree, model=model, prediction_method=method)
# Final = delta(x=ap, lambda0=lambda0, se=se, sim=sim, burn=burn, thin=thin, ent_type='LSE')
# print(Final)


''' ADDITIONAL INFORMATION

[1] Borges, R. et al. (2019). Measuring phylogenetic signal between categorical traits and phylogenies. Bioinformatics, 35, 1862-1869.
[2] Ishikawa, S. A. et al. (2019). A fast likelihood method to reconstruct and visualize ancestral scenarios. Molecular Biology and Evolution, 36, 2069-2085.
[3] Paradis, E. and Schliep, K. (2019). ape 5.0: an environment for modern phylogenetics and evolutionary analyses in R. Bioinformatics, 35, 526-528
[4] rpy2.github.io
[5] Joy,J.B. et al. (2016) Ancestral reconstruction. PLOS Computational Biology, 12. 
[6] Shannon,C.E. (1948) A mathematical theory of communication. Bell System Technical Journal, 27, 379-423. 
[7] Fitctree Growing Decision Trees - MATLAB & Simulink.

'''