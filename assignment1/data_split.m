function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data)
	data_size = size(data, 1);
	idx_shuffle = randperm( data_size );
	data_shuffle = data(idx_shuffle, :);
	
	feature_size = size(data, 2) - 1;
	X = data_shuffle(:, 1:feature_size);
  	X = feature_norm(X);
	Y = data_shuffle(:, feature_size + 1);

	train_size = floor(data_size / 10) * 7;
	val_size = floor(data_size / 10);
	test_size = data_size - train_size - val_size;
	
	X_train = X(1:train_size, :);
    Y_train = Y(1:train_size);

    X_val   = X(train_size+1:train_size+val_size, :);
    Y_val   = Y(train_size+1:train_size+val_size);

    X_test  = X(train_size+val_size+1:end, :);
    Y_test  = Y(train_size+val_size+1:end, :);

end
