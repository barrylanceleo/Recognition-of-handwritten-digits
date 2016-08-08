function [error_rate] = LR_test(data, target, k, theta_matrix)
%n is the number of samples and m is the number of features
[n, m] = size(data);
design_matrix = [ones(n, 1) data];

%target matrix
target_matrix = zeros(n, k);
for sample=1:length(target)
    target_matrix(sample, target(sample)+1) = 1;
end

%for multi-class we work with a soft max instead of a sigmoid function
%prediction vector for each class, a matrix of n x k dimension
A = design_matrix*theta_matrix;
exp_A = exp(A);
sum_exp_A = sum(exp_A, 2);
h_matrix = zeros(n, k);
for class=1:k
    h_matrix(:, class) = exp_A(:, class)./sum_exp_A; %softmax activation function
end

[prediction_probability predicted_class] = max(h_matrix,[],2);

%Classification error rate
Nwrong = 0;
for i=1:n
    if predicted_class(i) ~= target(i)+1
        Nwrong = Nwrong + 1;
    end
end

error_rate = Nwrong / n;

