%
%
%This is just a sample code (uncompleted) and you can implement the assignment from scrach.
%But please only print out the information listed (line 50, 63, 64, 65) in this sample code
%when submitting your code.
%
%
%If you use this sample structure, you need to implement the functions
%"fminunc_train", "logisticR_train", "logisticR_predict" 
%in the corresponding files by yourself. 
%The function "data_split" is provided as an example. You can rewrite it.
%
%
function main(data_name, model_name) 
    %data_name: Loading data - previously arg_list{1}
    %model_name: Identify the model name - previously arg_list{2}
    
    %Split the data into training, validation and test data sets, X is feature, Y is label
    %Some data may need normalization on the features
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data_name);

    %Weight initialization
    feature_size = size(X_train, 2);         %The first feature equals 1 for all instances.
    weights = randn(feature_size, 1) * 0.05; %Bias is integrated in weights. 
    %bias = randn(1);


    if strcmp(model_name, 'fminunc')
        %Training
        [loss_train, weights] = fminunc_train(X_train, Y_train, weights);
        %Evauate on validation data set
        accuracy_val = logisticR_predict(X_val, Y_val, weights);
        %Evaluate on testing data set
        accuracy_test = logisticR_predict(X_test, Y_test, weights);


    elseif strcmp(model_name, 'batch')
        iterations = 100;
        for i = 1:1:iterations
            %Training		 
            [loss_train, weights] = logisticR_train(X_train, Y_train, weights);		
            if i <= 10
                printf('Iteration %d loss: %f\n', i, loss_train);
            end
            %Evauate on validation data set
            accuracy_val = logisticR_predict(X_val, Y_val, weights);
        end
        %Evaluate on testing data set
        accuracy_test = logisticR_predict(X_test, Y_test, weights);
    else
        printf('Training model should be provided !!!!! \n')
        return
    end


    printf('Final loss for training data: %f\n', loss_train);
    printf('Final accuracy for validation data: %f\n', accuracy_val);
    printf('Final accuracy for test data: %f\n', accuracy_test);
end