function [J, grad] = costFunction(theta, X, y)
% ex 2
% COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
J = 0;
h_predictions = sigmoid(X*theta);
log_errors = -y.*log(h_predictions) - (1-y).*(log(1 - h_predictions));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%for iter = 1:m
%J=J+()^2;
%end
J = sum(log_errors)/m;


%J(?) =
%1
%m
%m
%X
%i=1
%? -y (i)
%log(h ? (x (i) )) - (1 - y (i) )log(1 - h ? (x (i) )) ? 


%?J(?)
%?? j
%=
%1
%m
%m
%X
%i=1
%(h ? (x (i) ) - y (i) )x (i)

xarg_1 = X(:,1); % Just 1-s for all rows, but for 
xarg_2 = X(:,2);
xarg_3 = X(:,3);

predictions = sigmoid(X*theta); % this is just h_{\theta}(x)

errors = predictions-y; 

errors_x1 = errors.*xarg_1;
errors_x2 = errors.*xarg_2;
errors_x3 = errors.*xarg_3;

grad(1,1) = 1/m*sum(errors_x1);
grad(2,1) = 1/m*sum(errors_x2);
grad(3,1) = 1/m*sum(errors_x3);


% =============================================================

end
