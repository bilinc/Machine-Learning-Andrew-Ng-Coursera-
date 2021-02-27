function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m i 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



# loop over centroids
##for i = 1:length(X)
##  init_dist = 1000;
##  # loop over examples
##  for k = 1:K
##    min_dist = (norm(X(i, :) - centroids(k, :), 2))^2;
##    
##    if min_dist < init_dist
##      idx(i) = k;
##      init_dist = min_dist;
##    endif
##    
##  endfor
##  
##endfor

% Alternative solution
My = zeros(size(X,1), K);   
for k = 1:K
  % vectorized solution to the euclidean norm. My is a [m x K] matrix with the ||x-u||^2 in each row
  My(:,k) = sqrt(sum((X-centroids(k,:)).^2, 2));    
endfor

% finds the row wise minimum value (2 indicates row wise, 1 is column wise)
[m, idx] = min(My, [], 2);
 
% =============================================================

end
