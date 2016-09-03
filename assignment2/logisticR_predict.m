function accuracy = logisticR_predict(X, Y, weights)
    n = size(X,1);
    res = zeros(n,1);
    for i = 1:n
        sigm = sigmoid(X(i,:) * weights);
        if sigm >= 0.5
            res(i) = 1;
        else
            res(i) = -1;
        end
    end

    matching = res == Y;
    matched = sum(matching);
    accuracy = matched / size(X, 1);
end