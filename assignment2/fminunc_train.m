function [loss, weights] = fminunc_train(X, Y, weights)

    df = @(w)(sum(X' * (Y - (1 / (1 + exp(sum(X * w)))))));   
    
	[weights, loss] = fminunc(df, weights);

end
