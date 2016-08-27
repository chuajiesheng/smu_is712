function [loss, weights, bias] = analytical_train(X, Y, weights, bias)
	
    % weight = [ (x^T * x)^-1 * x^T] * y
 	weights = (X.' * X) \ X.' * Y; 
    predict = X * weights + bias;
    loss = mean2((predict - Y).^2 / 2);
end
