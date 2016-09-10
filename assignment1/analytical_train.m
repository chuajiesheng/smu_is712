function [loss, weights, bias] = analytical_train(X, Y, ~, ~)
	
	data_size = size(X,1);
	feature_size = size(X,2);

	X_new = [X, ones(data_size, 1)];

	W = inv(X_new' * X_new) * X_new' * Y;

	weights = W(1:feature_size, 1);
	bias = W(feature_size + 1, 1);

	output = X * weights + bias;
	loss_minus = output - Y;

	loss = mean( loss_minus.^2 ) / 2;
end
