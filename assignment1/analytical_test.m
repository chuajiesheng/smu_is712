function loss = analytical_test(X, Y, weights, bias)

    predict = (weights.' * X.').' + bias
    loss = mean2((predict - Y).^2 / 2)
    
end
