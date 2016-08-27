function loss = analytical_test(X, Y, weights, bias)

    predict = (X * weights + bias)
	loss    = sum((predict - Y).^2)

end
