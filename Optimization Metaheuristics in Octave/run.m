function run()
  # Loading necessary packages
  pkg load statistics;

  # General parameters
  npop = 100;
  nvar = 20;
  ngen = 500;
  lb = zeros(nvar, 1);
  ub = ones(nvar, 1);

  # Particle Swam Optimization
  pso_weights = [1/6; 2/6; 2/3];
  tic;
  [x_pso, f_pso] = PSO(npop, nvar, ngen, pso_weights, lb, ub);
  toc;
  disp('PSO:');
  disp(strcat('Solution = ', mat2str(x_pso)));
  disp(strcat('Fitness = ', mat2str(f_pso)));
  disp('');

  # Continous Ant Colony Optimization
  L = nvar * 2;
  q = 0.3;
  csi = 0.85;
  tic;
  [x_caco, f_caco] = CACO(npop, nvar, ngen, L, q, csi, lb, ub);
  toc;
  disp('CACO:');
  disp(strcat('Solution = ', mat2str(x_caco)));
  disp(strcat('Fitness = ', mat2str(f_caco)));
  disp('');
end
