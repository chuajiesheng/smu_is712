function [loss, weights, bias] = analytical_train(X, Y, weights, bias)
	
    % weight = [ (x^T * x)^-1 * x^T] * y
 	weights = ((X.' * X) \ X.') * Y; 
	bias    = bias;
    predict = (X * weights) - Y + bias
	loss    = sum((predict - Y).^2)

end
