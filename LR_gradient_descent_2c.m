function [theta, J_history] = LR_gradient_descent_2c(data, target, eta, num_iters)

%n is the number of samples and m is the number of features
[n, m] = size(data);
design_matrix = [ones(n, 1) data];
theta = zeros(m+1, 1);
J_history = zeros(num_iters, 1);

for i=1:length(target)
    if target(i) == 0
        target(i) = 1;
    else
        target(i) = 0;
    end
end

for iter=1:num_iters
    %prediction after applying logistic funtion
    %h = sigmf(design_matrix*theta, [1 0]);
    h = 1./(1+exp(-(design_matrix*theta)));
    %cost function is -sum(target*log(h)+(1-target)*log(1-h))
    J = -sum(target.*log(h)+(1-target).*log(1-h)); 
    display(J);
    J_history(iter) = J; 
    gradient_J = design_matrix' * (h - target); 
    theta = theta - eta*gradient_J;
end