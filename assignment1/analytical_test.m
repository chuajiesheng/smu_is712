function loss = analytical_test(X, Y, weights, bias)

    predict = X * weights + bias;
    loss = mean((predict - Y).^2 / 2);

end
