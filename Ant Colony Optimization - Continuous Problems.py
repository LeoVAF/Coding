import numpy as np

class StoppingAlgorithm(Exception):
    """ Exception to stop the algorithm. """
    pass

class ACOR:
  def __init__(self, f, n_dim, m_ants, L, xmin, xmax, q=1e-4, csi=0.85, max_eval=1000, random_state=None):
    ''' Ant Colony Optimization for continuous problems. '''

    self.f = f # Objective function to minimize
    self.n_dim = n_dim # Number of dimensions
    self.m_ants = m_ants # Number of ants
    self.L = L # Archive size
    self.q = q # Locality of the process search parameter
    self.csi = csi # Evaporation coefficient
    self.xmin = xmin # Lower bound of the search space
    self.xmax = xmax # Upper bound of the search space
    self.max_eval = max_eval # Maximum number of evaluations

    self.X = np.empty((self.m_ants, self.n_dim)) # Position matrix of the ants
    self.F = np.empty(self.m_ants) # Fitness matrix of the ants

    self.eval_counter = 0 # Evaluation counter
    self.pheromone = np.zeros((L, len(xmin))) # Pheromone matrix
    self.pheromone_fit = np.full(L, np.inf) # Pheromone fitnesses
    self.weights = np.exp(-(np.arange(L)**2)/(2*((q*L)**2)))/(q*L*np.sqrt(2*np.pi)) # Weights for the pheromone matrix
    self.weights /= np.sum(self.weights) # Normalize weights
    self.best_solution = None  # Best solution found so far
    self.best_fitness = np.inf # Best objective value found so far

    # Set random seed for reproducibility
    if random_state is not None:
      np.random.seed(random_state)

    # Validate input parameters
    if not isinstance(xmin, np.ndarray) or not isinstance(xmax, np.ndarray):
      raise TypeError("xmin and xmax must be numpy arrays.")
    if xmin.shape != xmax.shape:
      raise ValueError("xmin and xmax must have the same shape.")
    if self.n_dim != len(xmin) or self.n_dim != len(xmax):
      raise ValueError("Dimensions of xmin and xmax must match n.")
    if self.n_dim <= 0 or self.m_ants <= 0 or self.L <= 0:
      raise ValueError("n, m, and L must be positive integers.")
    if self.q <= 0 or self.csi <= 0 or self.csi > 1:
      raise ValueError("q must be positive and csi must be in (0, 1].")
    if self.max_eval <= 0:
      raise ValueError("max_eval must be a positive integer.")
    if np.any(xmin > xmax):
      raise ValueError("xmin must be less than xmax.")
    if n_dim > L:
      raise ValueError("The number of archive solutions must be equal or greater than the number of dimensions.")

  def initialize(self):
    ''' Initialize the pheromone matrix. '''

    # Initialize pheromone matrix randomly within the bounds
    self.pheromone = np.random.uniform(self.xmin, self.xmax, (self.L, self.n_dim))
    # Initialize the fitness matrix of the pheromone
    self.pheromone_fit = self.evaluate(self.pheromone)
    # Sort the pheromone matrix based on fitness
    sorted_indices = np.argsort(self.pheromone_fit)
    self.pheromone = self.pheromone[sorted_indices, :]
    self.pheromone_fit = self.pheromone_fit[sorted_indices]
    # Get the best solution and fitness
    self.best_solution = self.pheromone[0]
    self.best_fitness = self.pheromone_fit[0]

  def evaluate(self, X):
    ''' Evaluate the fitness of the solutions in X. '''

    # Check if the stopping criterion reached
    if self.eval_counter >= self.max_eval:
        raise StoppingAlgorithm()
    # Get the size of the position matrix
    X_size = len(X)
    # Calculate the minimum number of fitness evaluations
    min_evaluations = min(self.max_eval - self.eval_counter, X_size)
    # Update the fitness counter
    self.eval_counter += min_evaluations
    # Evaluate the fitness function
    if(min_evaluations < X_size):
        # Evaluate the sliced particle positions for the minimum evaluations and concatenate the rest with np.inf
        return np.concatenate((np.array([self.f(x) for x in X[:min_evaluations]]), np.full(X_size - min_evaluations, np.inf)), axis=0)
    else:
        return np.array([self.f(x) for x in X])

  def update_pheromone(self):
    ''' Update the pheromone matrix. '''

    # Concatenate the pheromone matrix and the fitness values with the new solutions
    concatenated_fit = np.concatenate((self.pheromone_fit, self.F), axis=0)
    concatenated_pheromone = np.concatenate((self.pheromone, self.X), axis=0)
    # Sort the pheromone matrix based on fitness
    sorted_indices = np.argpartition(concatenated_fit, self.L)[:self.L]
    self.pheromone[:, :] = concatenated_pheromone[sorted_indices, :]
    self.pheromone_fit[:] = concatenated_fit[sorted_indices]

  def generate_solutions(self):
    ''' Generate solutions with ants using the pheromone matrix. '''

    # Get the indices to select the Gaussian distribution
    idxs = np.random.choice(self.L, size=(self.m_ants, self.n_dim), p=self.weights)
    # Get the averages of the selected pheromone solutions
    averages = self.pheromone[idxs, np.arange(self.n_dim)]
    # Get the standard deviations of the selected pheromone solutions
    std_devs = (self.csi/(self.L - 1)) * np.sum(np.abs((self.pheromone[np.newaxis, :, :] - averages[:, np.newaxis, :])), axis=1)
    ################# Traditional standard deviation calculation #################
    # std_devs__ = np.std(self.pheromone[np.newaxis, :, :], axis=1, ddof=1, mean=averages[:, np.newaxis, :])
    ##############################################################################
    # Generate the solutions using the Gaussian distribution
    solutions = np.random.normal(averages, std_devs, (self.m_ants, self.n_dim))
    # Clip the solutions to be within the bounds
    solutions = np.clip(solutions, self.xmin, self.xmax)
    return solutions

  def run(self):
    self.initialize()
    try:
      while True:
        # Generate new solutions using the pheromone matrix
        self.X[:, :] = self.generate_solutions()
        # Evaluate the fitness of the generated solutions
        self.F[:] = self.evaluate(self.X)
        # Sort the solutions based on fitness
        self.update_pheromone()
        # Get the best solution and best fitness
        self.best_solution = self.pheromone[0]
        self.best_fitness = self.pheromone_fit[0]
    except StoppingAlgorithm:
      return self.best_solution, self.best_fitness

##################################### RUNNING EXAMPLE BELOW #####################################

# Fitness functions to minimize
def select_function(option):
  if option.lower() == 'sphere':
    return lambda x: np.sum(x**2)
  elif option.lower() == 'rastrigin':
    return lambda x: 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
  elif option.lower() == 'griewank':
    return lambda x: 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
  elif option.lower() == 'ackley':
    return lambda x: -20*np.exp(-0.2*np.sqrt((np.sum(x**2))/len(x))) - np.exp((np.sum(np.cos(2*np.pi*x))/len(x))) + np.e + 20
  elif option.lower() == 'rosenbrock':
    return lambda x: np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
  else:
    raise ValueError("Invalid function name. Choose from 'sphere', 'rastrigin', 'griewank', 'ackley' or rosenbrock.")

# Setting parameters
n_dim = 10
m_ants = 20
L = n_dim + 5
q = 0.4
csi = 0.85
max_eval = 1000
xmin = np.array([-33]*n_dim)
xmax = np.array([33]*n_dim)

ant_colony = ACOR(select_function('ackley'), n_dim=n_dim, m_ants=m_ants, L=L, xmin=xmin, xmax=xmax, q=q, csi=csi, max_eval=max_eval)

x, fit = ant_colony.run()

print('Best solution:',x)
print('Best fitness:', fit)