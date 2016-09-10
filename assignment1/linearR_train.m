function [loss, weights, bias] = linearR_train(i, X, Y, weights, bias)
  learning_rate = 0.3;
  feature_size = size(weights, 1);

  output = X * weights + bias;
  loss_minus = output - Y;
  loss = mean( loss_minus.^2 ) / 2;

  weights_grad = mean(X .* repmat(loss_minus, 1, feature_size));
  bias_grad = mean(loss_minus);

  weights = weights - learning_rate * weights_grad';
  bias = bias - learning_rate * bias_grad;
end
