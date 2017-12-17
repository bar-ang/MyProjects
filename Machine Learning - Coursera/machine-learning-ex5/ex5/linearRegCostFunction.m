function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%





%calculating J
h_x = X*theta;
J = (h_x-y)'*(h_x-y);
reg = theta(2:end)'*theta(2:end) * lambda;
J = (J + reg)/(2*m);

%calculating the gradient
h_x = X*theta;
grad = X'*(h_x - y);
gradreg = lambda*theta;
gradreg(1) = 0;
grad = (grad + gradreg)/m;

% =========================================================================

grad = grad(:);

end
