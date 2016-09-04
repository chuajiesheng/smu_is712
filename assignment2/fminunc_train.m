function [loss, weights] = fminunc_train(X, Y, weights)

    df = @(w)(sum(X' * (Y - (1 / (1 + exp(sum(X * w)))))));   
    
    options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
	[weights, loss] = fminunc(df, weights, options);

end
