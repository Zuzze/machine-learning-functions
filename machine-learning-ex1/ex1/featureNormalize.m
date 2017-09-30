function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
%

% How many features (xs) there are in the data set?
% go through columns (second dimension (row x col <=> n x m) of X is 2
for i = 1:size(X,2) %

    % mean of all rows in col i
    mu(i) = mean(X(:,i));

    % numeretor of normalization (data value - mean value)
    X_norm(:,i) = X_norm(:,i) - mu(i);

    % denominator of normalization
    sigma(i) = std(X_norm(:,i));

    % normalized matrix X = (dataVal - meanVal) / standard deviation
    X_norm(:,i) = X_norm(:,i) / sigma(i);
end

% ============================================================

end
