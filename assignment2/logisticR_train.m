function [loss, weights] = logisticR_train(t, X, Y, weights)
    n = 1 / sqrt(t);
    
    [nSamples, nFeature] = size(X);
    
    temp = zeros(nFeature, 1);
    for i = 1:nSamples
        temp = temp + X(i,:)' * (Y(i) - sigmoid(X(i,:) * weights));
        weights = weights + n * temp;
    end
    
    loss = logisticR_predict(X, Y, weights);
end
