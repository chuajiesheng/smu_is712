function loss = analytical_test(X, Y, weights, bias)

    predict = (weights.' * X.').' + bias
    loss = sum((Y - predict).^2)
    
end
