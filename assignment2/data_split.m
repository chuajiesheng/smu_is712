function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data_name)
	if strcmp(data_name, "heart")
		data = importdata("../data/heart.dat");
		data(:, end) =  2 .* ( data(:, end) == 2 ) - 1;
		data_size = size(data, 1);
		idx_shuffle = randperm( data_size );
		data_shuffle = data(idx_shuffle, :);
	
		feature_size = size(data, 2) - 1;
		X = data_shuffle(:, 1:feature_size);
	
		X = feature_norm(X);

		X = [ones(data_size,1), X];
	
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
	elseif strcmp(data_name, "gisette")
		X_train = importdata("../data/gisette_train.data");
		X_train = [ones(size(X_train,1),1), X_train];
		X_train = X_train / 1000;
		Y_train = importdata("../data/gisette_train.labels");
		
		X = importdata("../data/gisette_valid.data");
		X = [ones(size(X,1),1), X];
		X = X / 1000;
		Y = importdata("../data/gisette_valid.labels");
		
		val_num = floor(size(X,1)/2);
		X_val = X(1:val_num, :);
		Y_val = Y(1:val_num, :);
		X_test = X(val_num+1:end, :);
		Y_test = Y(val_num+1:end, :);
	end
	

end
