%load images
train_data = loadMNISTImages('train-images.idx3-ubyte');
train_data = train_data';
test_data = loadMNISTImages('t10k-images.idx3-ubyte');
test_data = test_data';

%load labels
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

