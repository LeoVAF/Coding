function [X, F] = run()
  npop = 100;
  nvar = 2;
  ngen = 500;
  w = [1/6; 2/6; 2/3];
  lb = [0; 0];
  ub = [1; 1];
  
  tic;
  [X, F] = PSO(npop, nvar, ngen, w, lb, ub);
  toc;
end