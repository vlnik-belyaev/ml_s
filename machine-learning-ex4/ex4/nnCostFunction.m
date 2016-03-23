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
Theta1_grad = zeros(size(Theta1));  % 25*401
Theta2_grad = zeros(size(Theta2));  % 10*26

% ====================== YOUR CODE HERE ======================
% Prediction part (forward propagation)% input layera_1 = [ones(m, 1) X]; % 5000x401% hiden layerz_2 = Theta1*a_1';    % 25x401 * 401x5000 = 25x5000a_2 = sigmoid(z_2);   % 25x5000a_20 = [ones(m, 1) a_2']; % 5000x26% output layerz_3 = Theta2*a_20'; % 10x26 * 26x5000 = 10 x 5000;a_3 = sigmoid(z_3); % 10x 5000% prediction itself:h_theta = a_3;       % 10x 5000
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.myTrainedValuesAsColumns = zeros(num_labels,m);for i = 1:m  yTrainedValuesAsColumns(:,i) = yLabelToYVector(y(i,1), num_labels);end  costMatrix = -yTrainedValuesAsColumns.*log(h_theta)-(1-yTrainedValuesAsColumns).*log(1-h_theta); % 10x 5000
J = 1./m*sum(sum(costMatrix,1),2)
%
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
%
g_prime_z2 = a_20'.*(1-a_20'); % 26x5000 

for t = 1 : m	      % step 1 already done in forward propagation
	      % step 2:
	      y_t =  yTrainedValuesAsColumns(:,t);
	      delta_t3 = a_3(:,t) - y_t;                  % 10 x 1
	      % step 3
	      %delta_t2 = Theta2'*delta_t3.*sigmoidGradient(z2(:,t)); % 26x10 * 10x1 .*
				delta_t2 = Theta2'*delta_t3.*g_prime_z2(:,t); % 26x10 * 10x1 .* 26x1 = 26x1
        delta_t2 = delta_t2(2:end); % skip 
        % step 4
        a_2t = [1;a_2(:,t)]';                                      % 1x26 - hidden layer with bias unit
        Theta2_grad = Theta2_grad + delta_t3 * a_2t;               %10*26 = 10x1 * 1x26
        Theta1_grad = Theta1_grad + delta_t2 * a_1(t,:);           % 25x1 * 1x401 = 25*401
        
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_trimmed = Theta1(:,2:end);
Theta2_trimmed = Theta2(:,2:end);

J = J + lambda/(2*m)*(sum(sum(Theta1_trimmed.^2,1),2) + sum(sum(Theta2_trimmed.^2,1),2));

Theta1_zero_bias = [zeros(hidden_layer_size,1) Theta1_trimmed];
Theta2_zero_bias = [zeros(num_labels,1) Theta2_trimmed];


Theta1_grad_regular = (Theta1_grad + lambda*Theta1_zero_bias)/m;
Theta2_grad_regular = (Theta2_grad + lambda*Theta2_zero_bias)/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:)/m ; Theta2_grad(:)/m];
grad = [Theta1_grad_regular(:); Theta2_grad_regular(:)];

end
