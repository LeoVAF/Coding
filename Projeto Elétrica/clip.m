function X_clipped = clip(X, Xmin, Xmax)
  X_clipped = min(max(X, Xmin), Xmax);
endfunction