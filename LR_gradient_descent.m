function [theta_matrix, J_history] = LR_gradient_descent(data, target, eta, k, num_iters)

%n is the number of samples and m is the number of features
[n, m] = size(data);
design_matrix = [ones(n, 1) data];
J_history = zeros(num_iters, 1);

%target matrix
target_matrix = zeros(n, k);

%parameter matrix
theta_matrix = zeros(m+1, k);

for sample=1:length(target)
    target_matrix(sample, target(sample)+1) = 1;
end

for iter=1:num_iters
    %for multi-class we work with a soft max instead of a sigmoid function
    %prediction vector for each class, a matrix of n x k dimension
    A = design_matrix*theta_matrix;
    exp_A = exp(A);
    sum_exp_A = sum(exp_A, 2);
    h_matrix = zeros(n, k);
    for class=1:k
        h_matrix(:, class) = exp_A(:, class)./sum_exp_A; %softmax activation function
    end
    
    %h = 1./(1+exp(-(A))); %sigmoidal activation function
    %cost function is -sum(target*log(h)+(1-target)*log(1-h))
    J = -sum(sum(target_matrix.*log(h_matrix), 2));
    J_history(iter) = J; 
    gradient_J_matrix = design_matrix' * (h_matrix - target_matrix); 
    theta_matrix = theta_matrix - eta.*gradient_J_matrix;
    
    %error_rate
    [prediction_probability predicted_class] = max(h_matrix,[],2);
    Nwrong = 0;
    for i=1:n
        if predicted_class(i) ~= target(i)+1
            Nwrong = Nwrong + 1;
        end
    end
    fprintf('LR I: %d J: %f E: %f\n', iter, J, Nwrong/n);
end

display('end');