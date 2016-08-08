function [error_rate] = NN_test(data, target, k, num_hidden_units, theta_matrix_h, theta_matrix_o)

%n is the number of samples and m is the number of features
[num_inputs, m] = size(data);
design_matrix = [ones(num_inputs, 1) data];

%target matrix
target_matrix = zeros(num_inputs, k);
for sample=1:length(target)
    target_matrix(sample, target(sample)+1) = 1;
end

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

%error_rate
[prediction_probability predicted_class] = max(h_matrix,[],2);
Nwrong = 0;
for i=1:num_inputs
    if predicted_class(i) ~= target(i)+1
        Nwrong = Nwrong + 1;
    end
end

error_rate = Nwrong/num_inputs;