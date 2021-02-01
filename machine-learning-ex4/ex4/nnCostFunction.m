function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
% Total number of possible labels
K = num_labels;

% Input layer
a1 = [ones(m,1) X];     % Create first activation unit. Also add the bias unit
##size(a1)

% Hidden layer
z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];

##size(a2)

% Output layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

h = a3;

y_matrix = eye(num_labels)(y,:);  % create an eye matrix representing the labels in y
                                  % then generates a new matrix by "indexing" using the values in the y matrix columnwise
                                  % this creates a new matrix of another dimension

J = 1/m * sum(sum(-y_matrix.*log(h)-(1-y_matrix).*log(1-h)));

% Regularized cost function

% Don't regularize the bias unit
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Step 2: error vector for final layer L (L = 3)
y_logical = [];

for c = 1:num_labels;
  lgl = (y==c);
  y_logical = [y_logical (y==c)];
end

dL = a3 - y_logical;

% Step 3: error vector for all other layers (total L-1 layers)
g_prime = sigmoidGradient(z2);
d2 = dL*Theta2(:, 2:end).*g_prime;    % if you compute the sigmoid gradient of z2, 
%                                       then you must exclude the first column of Theta2 when you compute d2

% Step 4: accumulate the gradient
Delta2 = dL'*a2;
Delta1 = d2'*a1;

% Step 5: obtain the unregularized gradient
Theta2_grad = 1/m * Delta2;
Theta1_grad = 1/m * Delta1;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad(:, 2:end) += lambda/m*Theta2(:, 2:end);
Theta1_grad(:, 2:end) += lambda/m*Theta1(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
