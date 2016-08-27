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

%Command line parameters
arg_list = argv ();

%Loading data
file_path = strcat("../data/", arg_list{1});
data = importdata(file_path);

%Identify the model name
model_name = arg_list{2};

%Split the data into training, validation and test data sets, X is feature, Y is label
%Some data may need normalization on the features
[X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data);

%Weight initialization
feature_size = size(X_train, 2);
weights = randn(feature_size, 1) * 0.5;
bias = randn(1);


if strcmp(model_name, "analytical")
	%Training
	[loss_train, weights, bias] = analytical_train(X_train, Y_train, weights, bias);
	%Evauate on validation data set
	loss_val   = analytical_test(X_val, Y_val, weights, bias);
	%Evaluate on testing data set
	loss_test  = analytical_test(X_test, Y_test, weights, bias);
	

elseif strcmp(model_name, "iterative")
	iterations = 1000;
	for i = 1:1:iterations
		%Training		 
		[loss_train, weights, bias] = linearR_train(X_train, Y_train, weights, bias);
		if i <= 10
		    printf("Iteration %d loss: %f\n", i, loss_train);
		end
		%Evauate on validation data set
		loss_val = linearR_predict(X_val, Y_val, weights, bias);
	end
	%Evaluate on testing data set
	loss_test = linearR_predict(X_test, Y_test, weights, bias);
else
	printf("Training model should be provided !!!!! \n")
	return
end


printf("Final loss for training data: %f\n", loss_train);
printf("Final loss for validation data: %f\n", loss_val);
printf("Final loss for test data: %f\n", loss_test);
