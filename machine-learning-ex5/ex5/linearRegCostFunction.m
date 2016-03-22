function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
theta_size = size(theta,1) % number of theta components
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

prediction = X*theta;

J = 1/(2*m)*sum((prediction-y).^2) + lambda/(2*m)*sum(theta(2:end).^2);



grad_1 = 1/m*sum(prediction-y);
% Try to make it general - vectorized form
%grad_2 = 1/m*sum((prediction-y).*X(:,2)) + lambda/m*theta(2,1);
grad_2_end = 1/m*sum((prediction-y).*X(:,2:end),1)+ lambda/m*theta(2:end,1)';

%grad = [grad_1;grad_2];
grad = [grad_1;grad_2_end'];

% =========================================================================

grad = grad(:);

end
