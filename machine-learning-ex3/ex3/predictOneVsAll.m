function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

% size of dimension 1 (row) of matrix X
m = size(X, 1);

% rows of theta vector
num_labels = size(all_theta, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the
%       max element, for more information see 'help max'. If your examples
%       are in rows, then, you can use max(A, [], 2) to obtain the max
%       for each row.
%

% for each row in X
for i = 1:m

    % X(i,:) is ith row vector in X e.g. [1 2 3]
    % theta is e.g. [3; 4; 5] --> num_labels = 3
    % repeat current matrix row on top of each other num_labels (=size of theta matrix) times
    RX = repmat(X(i,:),num_labels,1);
    % in the example RX = [1 2; 1 2; 1 2]

    % multiply repeated matrix for each element with theta
    RX = RX .* all_theta;
    % in the example RX = []

    % sums of second dimensions (rows, RX) to one matrix SX
    SX = sum(RX,2);

    % the row in the matrix with largest sum
    [val, index] = max(SX);
    p(i) = index;
end





% =========================================================================


end
