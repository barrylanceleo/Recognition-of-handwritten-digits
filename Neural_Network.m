eta = 0.00001;
num_iters = 10000;
num_classes = 10;
num_hidden_units = 100;

%perform gradient descent
[theta_matrix_h, theta_matrix_o, J_history] = NN_gradient_descent(train_data, train_labels, eta, num_classes, num_hidden_units, num_iters);

%get the misclassification rate
[error_rate] = NN_test(test_data, test_labels, num_classes, num_hidden_units, theta_matrix_h, theta_matrix_o);

fprintf('NN Error rate: %f.\n', error_rate);

%save the parameters
Wnn1 = theta_matrix_h(2:size(train_data, 2)+1, :);
bnn1 = theta_matrix_h(1, :);
Wnn2 = theta_matrix_o(2:num_hidden_units+1, :);
bnn2 = theta_matrix_o(1, :);
h = 'sigmoid';
save proj3.mat Wlr blr Wnn1 bnn1 Wnn2 bnn2 h