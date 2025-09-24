function [x, f] = CACO(npop, # Number of ants
                       nvar, # Number of variables
                       ngen, # Number of iterations
                       L,    # Archive size (L >= nvar)
                       q,    # Locality of the process search parameter
                       csi,  # Evaporation coefficient
                       Xmin, # Variables' lower bound [x1_min; ...; xn_min]
                       Xmax) # Variables' upper_bound [x1_max; ...; xn_max]
  ############## Check the parameters ##############
  if L < nvar
    error('The parameter "L" must be greater than "nvar"!');
  endif
  if csi <= 0
    error('The parameter "csi" must be greater than 0!');
  endif
  if (nvar != length(Xmin)) || (nvar != length(Xmax))
    error('The number of variables must be the same length of the bound arrays "Xmin" and "Xmax"!');
  endif
  if any(Xmin > Xmax)
    error('The parameter "Xmin" must have all elements less than all compontnents in the "Xmax"!');
  endif
  ##################################################

  % ===================== Initialization =======================
  % Initial pheromone file (random solutions)
  pheromone = rand(nvar, L) .* (Xmax - Xmin) + Xmin;
  pheromone_fit = inf(L, 1);
  for i = 1:L # May use parfor
    pheromone_fit(i) = fobj(pheromone(:, i));
  end
  % Sort the initial file
  [pheromone_fit, idx] = sort(pheromone_fit);
  pheromone(:) = pheromone(:, idx);
  % Best initial solution
  x = zeros(nvar, 1); # Pre-allocating
  x(:) = pheromone(:, 1);
  f = pheromone_fit(1);
  % Weights (probability of selecting solutions in the file)
  k = (0:L-1)';
  weights = exp(-(k.^2) ./ (2*(q*L)^2)) ./ (q*L*sqrt(2*pi));
  weights = weights ./ sum(weights);
  % Precompute cumulative weights
  roulette = [0; cumsum(weights)];
  cumw = cumsum(weights);

  # Pre-allocating matrices
  r = zeros(nvar, npop);
  idxs = zeros(nvar, npop);
  row_idxs = repmat((1:nvar)', 1, npop);
  csi_constant = (csi/(L-1));
  mu = zeros(nvar, npop);
  sigma = zeros(nvar, npop);
  normal_vector = zeros(nvar, npop);
  all_solutions = zeros(nvar, npop + L);
  all_fitness = zeros(npop + L, 1);
  X = zeros(nvar, npop);
  F = zeros(npop, 1);
  % ====================== Main Loop =====================
  for gen = 1:ngen
    % ---------- Generation of new solutions ----------
    % Draw all random numbers
    r(:) = rand(nvar, npop);
    % Map to indices (roulette wheel selection)
    [~, idxs(:)] = histc(r, roulette);
    # Get linear indices from subscripts indices
    linear_idxs = sub2ind(size(pheromone), row_idxs, idxs);
    # Calculate the average
    mu(:) = pheromone(linear_idxs);
    # Calculate the standard deviation
    for j = 1:nvar # May use parfor
      sigma(j,:) = csi_constant * sum(abs(pheromone(j,:)' - mu(j,:)), 1);
    end
    # Sample the solutions
    normal_vector(:) = randn(nvar, npop);
    # generate the new solutions
    X(:) = mu + sigma .* normal_vector;
    % Clip the solutions
    X(:) = clip(X, Xmin, Xmax);
    % Evaluate new solutions
    for i = 1:npop # May use parfor
      F(i) = fobj(X(:, i));
    end
    % ---------- Evaluate file ----------
    all_solutions(:) = [pheromone X];
    all_fitness(:) = [pheromone_fit; F];
    [all_fitness(:), idx] = sort(all_fitness); # mink function is only better when we want to sort fewer than 5% of the elements
    all_solutions(:) = all_solutions(:, idx);
    % Keep only best L solutions
    pheromone(:) = all_solutions(:, 1:L);
    pheromone_fit(:) = all_fitness(1:L);
    % Update best solution
    if pheromone_fit(1) < f
      x(:) = pheromone(:, 1);
      f = pheromone_fit(1);
    end
  end
end
