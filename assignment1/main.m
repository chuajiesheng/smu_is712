%
%
%This is just a sample code (uncompleted) and you can implement the assignment from scrach.
%But please only print out the information listed (line 50, 63, 64, 65) in this sample code
%when submitting your code.
%
%
%If you use this sample structure, you need to implement the functions
%"analytical_train", "analytical_test", "linearR_train", "linearR_predict" 
%in the corresponding files by yourself. 
%The function "data_split" is provided as an example. You can rewrite it.
%
%
function main(file, model_name)
    fprintf('Starting\n');

    %Loading data
    file_path = strcat('../data/', file);
    fprintf('Using file: %s\n', file_path);
    data = importdata(file_path);
    
    %Identify the model name
    fprintf('Using model: %s\n', model_name);

    %Split the data into training, validation and test data sets, X is feature, Y is label
    %Some data may need normalization on the features
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data);

    %Weight initialization
    feature_size = size(X_train, 2);
    weights = randn(feature_size, 1) * 0.5;
    bias = randn(1);

    %Plot the training data
    plot(X_train, Y_train, 'o', 'MarkerFacecolor', 'r', 'MarkerSize', 8);
    
    if strcmp(model_name, 'analytical')
        %Training
        [loss_train, weights, bias] = analytical_train(X_train, Y_train, weights, bias);
        %Evauate on validation data set
        loss_val   = analytical_test(X_val, Y_val, weights, bias);
        %Evaluate on testing data set
        loss_test  = analytical_test(X_test, Y_test, weights, bias);


    elseif strcmp(model_name, 'iterative')
        iterations = 1000;
        for i = 1:1:iterations
            %Training		 
            [loss_train, weights, bias] = linearR_train(i, X_train, Y_train, weights, bias);
            if (iterations - i) <= 10
                fprintf('Iteration %d loss: %f\n', i, loss_train);
            end
            %Evauate on validation data set
            loss_val = linearR_predict(X_val, Y_val, weights, bias);
        end
        %Evaluate on testing data set
        loss_test = linearR_predict(X_test, Y_test, weights, bias);
    else
        printf('Training model should be provided !!!!! \n')
        return
    end


    fprintf('Final loss for training data: %f\n', loss_train);
    fprintf('Final loss for validation data: %f\n', loss_val);
    fprintf('Final loss for test data: %f\n', loss_test);
end