# Imports
import numpy as np
import os

# Delta functions
from delta_functs import delta

# ACE Method
def read_file_ace(file_path, separator=',', rmv_col_name=True, rmv_row_name=True):
    ''' file_path = Represents the path to the file that you want to read.
    separator     (default: ',')  = Specifies the separator used in the file to separate the values.
    rmv_col_name  (default: True) = Boolean value that determines whether the first row (column names) should be removed from the array.
    rmv_row_name  (default: True) = Boolean value that determines whether the first column (index) should be removed from the array.'''

    # Read the data from the file using numpy's genfromtxt function
    array = np.genfromtxt(file_path, delimiter = separator)
    
    # Check if the column name should be removed
    if rmv_col_name == True:
        array = np.delete(array, 0, axis=0)     # Remove the first row (Trait Name)
    
    # Check if the row name should be removed 
    if rmv_row_name == True:
        array = np.delete(array, 0, axis=1)     # Remove the first column (Entity Name)
        
    return array

def multiple_files_ap( dir=None, separator=',', rmv_col_name=True, rmv_row_name=True, lambda0=0.1, se=0.5, sim=100000, thin=10, burn=100, ent_type='LSE' ):
    '''dir       (default: None)   = Represents the directory path with the ancestral probabilities matrices. If no value is provided when calling the function, the user will be prompted to input the directory path.
    separator    (default: ',')    = Determines the separator used in the matrices when reading the files.
    rmv_col_name (default: True)   = Boolean value that indicates whether the column names should be removed when reading the files.
    rmv_row_name (default: True)   = Boolean value that indicates whether the row names should be removed when reading the files.
    lambda0      (default: 0.1)    = A constant value used in the acceptance ratio computations.
    se           (default: 0.5)    = The standard deviation used for the random walk in the Metropolis-Hastings algorithm.
    sim          (default: 100000) = The number of total iterations in the Markov Chain Monte Carlo (MCMC) simulation.
    thin         (default: 10)     = The thinning parameter, i.e., the number of iterations to discard between saved samples.
    burn         (default: 100)    = The number of burn-in iterations to discard at the beginning of the simulation.
    ent_type     (default: 'LSE')  = A string specifying the type of entropy calculation (options: 'LSE', 'SE', or any other value for Gini impurity).'''
    
    if dir == None:
        dir = str(input( 'What is the directory path with the ancestral probabilities matrices?' ))

    dic_files = set( os.listdir(dir) )
    dic_delta = set()
    for file in dic_files:
        file = dir + file
        ace = read_file_ace( file, separator, rmv_col_name, rmv_row_name )
        dic_delta.add( delta(x=ace, lambda0=lambda0, se=se, sim=sim, burn=burn, thin=thin, ent_type=ent_type) )
    
    return dict(zip( dic_files, dic_delta ))

# Delta Inputs
lambda0  = 0.1                       # rate parameter of the proposal
se       = 0.5                       # standard deviation of the proposal
sim      = 100000                    # number of iterations
thin     = 10                        # Keep only each xth iterate
burn     = 100                       # Burned-in iterates
ent_type = 'LSE'                     # Linear Shannon Entropy

# Path and Files
path_ap      = r"./input/Simplified/Ancestral_Probabilities/FILE"       # Path to the (ap: ancestral probabilities) input files
file         = "Simplified_AP_1.txt"                                    # File with Ancestral Probabilities
file         = path_ap.replace('FILE', file)                            # Path + file

# Calculate
Delta_Final_dic = multiple_files_ap( dir = './input/Simplified/Ancestral_Probabilities/' )

print( Delta_Final_dic )
print( Delta_Final_dic.values() )