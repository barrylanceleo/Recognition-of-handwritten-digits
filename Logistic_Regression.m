eta = 0.00001;
num_iters = 10000;
num_classes = 10;

%perform gradient descent
[theta_matrix, J_history] = LR_gradient_descent(train_data, train_labels, eta, num_classes, num_iters);

%get the misclassification rate
[error_rate] = LR_test(test_data, test_labels, num_classes, theta_matrix);

fprintf('Error rate: %f.\n', error_rate);

%save the parameters
Wlr = theta_matrix(2:size(train_data, 2)+1, :);
blr = theta_matrix(1, :);
save proj3.mat Wlr blr Wnn1 bnn1 Wnn2 bnn2 h