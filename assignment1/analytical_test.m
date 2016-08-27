function loss = analytical_test(X, Y, weights, bias)

    weights = ((X.' * X) \ X.') * Y; 
    predict = (X * weights) - Y + bias
	loss    = sum((predict - Y).^2)

end
