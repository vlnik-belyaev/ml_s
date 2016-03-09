function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1); % 10
num_hidden_layer = size(Theta1, 1); % 25
 
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% input layer

a_1 = [ones(m, 1) X]; % 5000x401

% hiden layer
z_2 = Theta1*a_1';    % 25x401 * 401x5000 = 25x5000
a_2 = sigmoid(z_2);   % 25x5000

a_20 = [ones(m, 1) a_2']; % 5000x26

% output layer
z_3 = Theta2*a_20'; % 10x26 * 26x5000 = 10 x 5000;
a_3 = sigmoid(z_3); % 10x 5000

[pmax, ind_pmax] = max(a_3', [], 2);
p = ind_pmax;
% =========================================================================


end
