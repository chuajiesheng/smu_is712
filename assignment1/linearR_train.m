function [loss, weights, bias] = linearR_train(i, X, Y, weights, ~)
    [~, n] = size(X);

    alpha = 1.0;
    alpha_i = alpha / (n * sqrt(i + 1));
    
    iterations = n;
    for j = 1:1:iterations
        Xj = X(j, :);
        gradient = sum((dot(weights', Xj) - Y(j)) * Xj);
        weights(j) = alpha_i * gradient;
    end
    
    bias = mean2(Y - X * weights);
    predict = X * weights + bias;
    loss = mean2((predict - Y).^2 / 2);
end
