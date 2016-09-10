function loss = analytical_test(X, Y, weights, bias)
  output = X * weights + bias;
  loss_minus = output - Y;
  loss = mean( loss_minus.^2 ) / 2;
end
