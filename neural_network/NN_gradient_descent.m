function [theta_matrix_h, theta_matrix_o, J_history] = NN_gradient_descent(data, target, eta, k, num_hidden_units, num_iters)

%n is the number of samples and m is the number of features
[num_inputs, m] = size(data);
design_matrix = [ones(num_inputs, 1) data];
J_history = zeros(num_iters, 1);

%target matrix
target_matrix = zeros(num_inputs, k);
for sample=1:length(target)
    target_matrix(sample, target(sample)+1) = 1;
end

%initialize parameter matrices
init_epsilon = .1;
%parameters for the hidden layer
theta_matrix_h = rand(m+1, num_hidden_units) * (2*init_epsilon) - init_epsilon;
%parameters for the output layer
theta_matrix_o = rand(num_hidden_units+1, k) * (2*init_epsilon) - init_epsilon;

for iter=1:num_iters

    %forward propogation to calculate the predictions at each layer
    %prediction vector for each class in the hidden layer, 
    %a matrix of num_inputs x num_hidden_units dimension 
    A_h = design_matrix*theta_matrix_h;
    z_h = 1./(1+exp(-(A_h)));  %using sigmoidal activation function in the hidden layer
    z_h = [ones(num_inputs, 1) z_h];
    
    %prediction vector for each class in the output layer, 
    %a matrix of num_inputs x k dimension 
    A_o = z_h*theta_matrix_o;
    exp_A_o = exp(A_o);
    sum_exp_A = sum(exp_A_o, 2);
    h_matrix = zeros(num_inputs, k);
    for class=1:k
        h_matrix(:, class) = exp_A_o(:, class)./sum_exp_A; %softmax activation function
    end
    
    %cost function
    J = -sum(sum(target_matrix.*log(h_matrix), 2));
    J_history(iter) = J; 
    
    %backward propogation to calculate the cost gradient at each layer
    delta_o = h_matrix - target_matrix;
    
    %differentiation of  activation function
    g_diff = z_h .* (1-z_h);
    
    delta_h = zeros(num_hidden_units+1, num_inputs);
    theta_o_delta = theta_matrix_o * delta_o';  %num_hidden_units+1 x num_inputs
    
    for input=1:num_inputs
        delta_h(:, input) = g_diff(input, :)' .*  theta_o_delta(:, input); 
    end
    
    gradient_theta_h = design_matrix' * delta_h(2:num_hidden_units+1, :)';
    gradient_theta_o = z_h' * delta_o;
    theta_matrix_h = theta_matrix_h - eta.*gradient_theta_h;
    theta_matrix_o = theta_matrix_o - eta.*gradient_theta_o;
        
    %error_rate
    [prediction_probability predicted_class] = max(h_matrix,[],2);
    Nwrong = 0;
    for i=1:num_inputs
        if predicted_class(i) ~= target(i)+1
            Nwrong = Nwrong + 1;
        end
    end
    fprintf('NN I: %d J: %f E: %f\n', iter, J, Nwrong/num_inputs);

end

display('end');