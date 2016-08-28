function [loss, weights, bias] = linearR_train(i, X, Y, weights, ~)
    [~, n] = size(X);

    alpha = 1.0;
    alpha_i = alpha / (n * sqrt(i + 1));
    
    for j = 1:1:n
        Xj = X(j, :);
        gradient = sum((dot(weights', Xj) - Y(j)) * Xj);
        weights(j) = alpha_i * gradient;
    end
    
    bias = mean(Y - X * weights);
    predict = X * weights + bias;
    loss = mean((predict - Y).^2 / 2);
end
