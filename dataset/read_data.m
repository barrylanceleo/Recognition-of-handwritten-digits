file_name = fopen('train-images.idx3-ubyte');
magic_num = swapbytes(fread(file_name, [1], '*uint32'));
train_num_images = swapbytes(fread(file_name, [1], '*uint32'));
num_rows = swapbytes(fread(file_name, [1], '*uint32'));
num_cols = swapbytes(fread(file_name, [1], '*uint32'));
train_data = cast(fread(file_name, [train_num_images, num_rows*num_cols], '*uint8'), 'double');
train_data = train_data./255;
fclose(file_name);

file_name = fopen('train-labels.idx1-ubyte');
magic_num = swapbytes(fread(file_name, [1], '*uint32'));
train_num_images = swapbytes(fread(file_name, [1], '*uint32'));
train_labels = cast(fread(file_name, '*uint8'), 'double');
fclose(file_name);

file_name = fopen('t10k-images.idx3-ubyte');
magic_num = swapbytes(fread(file_name, [1], '*uint32'));
test_num_images = swapbytes(fread(file_name, [1], '*uint32'));
num_rows = swapbytes(fread(file_name, [1], '*uint32'));
num_cols = swapbytes(fread(file_name, [1], '*uint32'));
test_data = cast(fread(file_name, [test_num_images, num_rows*num_cols], '*uint8'), 'double');
test_data = test_data./255;
fclose(file_name);

file_name = fopen('t10k-labels.idx1-ubyte');
magic_num = swapbytes(fread(file_name, [1], '*uint32'));
test_num_images = swapbytes(fread(file_name, [1], '*uint32'));
test_labels = cast(fread(file_name, '*uint8'), 'double');
fclose(file_name);

clear file_name magic_num num_rows num_cols ans  