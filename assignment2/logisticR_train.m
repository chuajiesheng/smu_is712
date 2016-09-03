function [loss, weights] = logisticR_train(t, X, Y, weights)
    n = 1 / sqrt(t);
    
    [nSamples, nFeature] = size(X);
    
    temp = zeros(nFeature, 1);
    for i = 1:nSamples
        temp = temp + (sigmoid(X(i,:) * weights) - Y(i)) * X(i,:)';
        weights = weights - n * temp;
    end
    
    loss = 1 - logisticR_predict(X, Y, weights);
end
