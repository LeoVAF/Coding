function [x, f] = PSO(npop,  # Number of particles
                      nvar,  # Number of variables
                      ngen,  # Number of generations
                      w,     # PSO's weights [w1; w2; w3]
                      Xmin,    # Variables' lower bound [x1_min; ...; xn_min]
                      Xmax)    # Variables' upper bound [x1_max; ...; xn_max]

  ############## Check the parameters ##############
  if (nvar != length(Xmin)) || (nvar != length(Xmax))
    error('The number of variables must be the same length of the bound arrays "Xmin" and "Xmax"!');
  endif
  if any(Xmin > Xmax)
    error('The parameter "Xmin" must have all elements less than all compontnents in the "Xmax"!');
  endif
  ##################################################

  # Set the maximum and minimum velocities
  Vmax = Xmax - Xmin;
  Vmin = -Vmax;
  # Set the initial position randomly
  X = rand(nvar, npop) .* Vmax + Xmin;
  # Set the initial velocity randomly
  V = rand(nvar, npop) .* (2 .* Vmax) + Vmin;
  # Calculate the initial particle fitnesses (it can be done with parfor too)
  F = zeros(npop, 1);
  for p=1:npop
    F(p) = fobj(X(:, p));
  endfor
  # Calculate the initial personal best (by copying data)
  Xpb = X(:, :);
  Fpb = F(:);
  # Calculate the global best (the solution of the problem)
  [fgb, argmin] = min(F);
  xgb = zeros(nvar, 1); # Pre-allocating
  xgb(:) = X(:, argmin);

  # Pre-allocating
  idx_mask = logical(zeros(npop, 1));
  rand1 = zeros(nvar, npop);
  rand2 = zeros(nvar, npop);
  # Main loop
  for g=1:ngen
    # Calculate the next particle velocities
    rand1(:) = rand(nvar, npop);
    rand2(:) = rand(nvar, npop);
    V(:) = w(1)*V + w(2)*rand1.*(Xpb - X) + w(3)*rand2.*(xgb - X);
    # Clip the particle velocities to the bounds
    V(:) = clip(V, Vmin, Vmax);
    # Calculate the next particle positions
    X += V;
    # Clip the particle positions
    X(:) = clip(X, Xmin, Xmax);
    ####### Update the information for the new generation #######
    # Evaluate the fitness function
    for p=1:npop
      F(p) = fobj(X(:, p));
    endfor
    # Check if the next positions are better than the previous positions
    idx_mask(:) = F < Fpb;
    # Update personal best
    Xpb(:, idx_mask) = X(:, idx_mask);
    Fpb(idx_mask) = F(idx_mask);
    # Update the global best
    [fmin, argmin] = min(F);
    if fmin < fgb
      fgb = fmin;
      xgb(:) = X(:, argmin);
    endif
   ###############################################################
  endfor
  x = xgb;
  f = fgb;
endfunction
