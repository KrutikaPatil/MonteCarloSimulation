import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal

def f_X(X): 
    '''
    This function contains the ODEs
    '''
    x = X[0,0]
    y = X[0,1]
    z = X[0,2]
    dx = 10 * ( y - x )
    dy = x * ( 28 - z ) - y
    dz = x * y - (8.0/3.0) * z
    return np.array([ dx, dy, dz ]).reshape(1,3)

def runge_kutta_4_solver(f, X_t, time_step = 0.05):
    '''
    This function has the runge kutta solver
    '''
    h = time_step
    k1 = f( X_t )
    k2 = f( X_t + 0.5 * k1 * h )
    k3 = f( X_t + 0.5 * k2 * h )
    k4 = f( X_t + k3 * h )
    X_t_plus = X_t + (1/6.0) * (k1 + 2*k2 + 2*k3 + k4) * h
    return X_t_plus


def simulate(f, X_0, duration, time_step = 0.05 ):
    '''
    Given initial conditions along with the ODEs this function simulates the ODEs for the specified duration
    The output is a numpy array with values of x,y,z at increasing time steps
    '''
    num_time_steps = int(duration / time_step)
    X_time_series = np.ones((num_time_steps,3))    
    X_time_series[0] = X_0    
    for i in range( num_time_steps-1):
        X_t = X_time_series[ i,:].reshape(1,3)
        X_t_plus = runge_kutta_4_solver(f_X, X_t, time_step = 0.05)
        X_time_series[i+1,:] = X_t_plus
        
    return X_time_series


def generate_samples(n_samples , mean_ = np.ones(3), covariance_matrix = np.eye(3)):
    '''
    This function generates a sample of initial conditions using mean and covariance
    '''
    random_samples = np.random.multivariate_normal( mean_, covariance_matrix, n_samples)
    print(random_samples.shape) # (100,3)
    return random_samples

if __name__ =='__main__':

    n_samples = 10 # no of initial conditions sampled
    time_step = 0.05 # time step for simulation
    duration = 15 # total duration of simulation
    num_time_steps = int(duration / time_step)
    sampled_initial_conditions = generate_samples(n_samples, mean_ = np.ones(3), covariance_matrix = np.eye(3)) # np array (10000,3)
    output_array = np.ones((n_samples, num_time_steps, 3)) # The simulation output will be stored in this array

    for sample_no in range(n_samples):
        print(sample_no)
        initial_condition = sampled_initial_conditions[sample_no].reshape(1,3)
        X_T = simulate(f_X, initial_condition, time_step = 0.05, duration = 15)
        output_array[sample_no] = X_T
    np.save('out',output_array) # simulation output is written to a file 
    # It is an np array with size 10,000 : number 0f initial conditions * 300 : number_of_simulation_time_steps * 3 : (x,y,z)
    


