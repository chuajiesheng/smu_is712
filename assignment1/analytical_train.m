function [loss, weights, bias] = analytical_train(X, Y, ~, ~)
	
    % weight = [ (x^T * x)^-1 * x^T] * y
  	weights = (X.' * X) \ X.' * Y; 
    bias = mean(Y - X * weights);
    predict = X * weights + bias;
    loss = mean((predict - Y).^2 / 2);
end
