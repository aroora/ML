function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%dataset of possible signma and C values
dataset = [0.01, 0.03, 0.1, 0.3,  1, 3, 10, 30];

% size of dataset
n = length(dataset);

%result vector
result = [];

for i=1:n
    for j=1:n
      
      C_temp = dataset(i);
      sigma_temp = dataset(j);
      
      model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
      predictions = svmPredict(model, Xval);
      
      error_temp = mean(double(predictions ~= yval));
      
      result = [result; C_temp, sigma_temp, error_temp];
    end
end

[error,index] = min(result(:,3));

C = result(index,1);
sigma = result(index,2);

% =========================================================================

end
