function [J, grad] = costFunctionReg(theta, X, y, lambda)
% ex 2 
% COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter 

J = 0;

h_predictions = sigmoid(X*theta);

log_errors = -y.*log(h_predictions) - (1-y).*(log(1 - h_predictions));

thetaLength = length(theta);

regularizedTheta = theta([2:thetaLength],1);

J = sum(log_errors)/m + lambda/(2*m)*sum(regularizedTheta.^2);

% Gradient computation

grad = zeros(size(theta));

featureCount = thetaLength;


predictions = sigmoid(X*theta); % this is just h_{\theta}(x)

errors = predictions-y;

% special case of fake first feature
xarg_1 = X(:,1); % Just 1-s for all rows,
errors_x1 = errors.*xarg_1;
grad(1,1) = 1/m*sum(errors_x1);

 for i = 2:featureCount
   xarg_i = X(:,i);
   errors_xi = errors.*xarg_i;
   grad(i,1) = 1/m*sum(errors_xi) + lambda/m*theta(i,1);
 end;
 

end
