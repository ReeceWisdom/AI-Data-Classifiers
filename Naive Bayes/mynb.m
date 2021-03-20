classdef mynb
    
    methods(Static)
        
        function m = fit(train_examples, train_labels)
            
            % array of all unique training labels
            m.unique_classes = unique(train_labels);
            % calculates the number of the unique labels
            m.n_classes = length(m.unique_classes);
            % mean & standard deviation used to generate the 
            % 'Probability Density' / 'Normal Distribution'
            m.means = {}; % empty cell array to store the mean
            m.stds = {}; % empty cell array to store the standard deviation
            
            % loops through all of the unique training labels
            for i = 1:m.n_classes
                
                % extracts first label from the array of unique labels
				this_class = m.unique_classes(i);
                % extracts all training examples which belong to that extracted label
                examples_from_this_class = train_examples{train_labels == this_class, :};
                % adds the mean of that label's training examples to the 'means' array
                m.means{end+1} = mean(examples_from_this_class);
                % adds the std of that label's training examples to the 'stds' array
                m.stds{end+1} = std(examples_from_this_class);
                
            end
            
            % empty cell array to store how often each label occurs 
            % in the all the training examples as a decimal
            % (Naive Bayes assumes dependency between true class label and observed features) 
            m.priors = []; 
            
            % loops through all of the unique training labels
            for i = 1:m.n_classes
                
                % extracts first label from the array of unique labels
				this_class = m.unique_classes(i);
                % extracts all training examples which belong to that extracted label
                examples_from_this_class = train_examples{train_labels == this_class, :};
                % calculates the number of the current label's training examples 
                % in comparison to the total number of training examples as decimal
                
                % the decimal of that label's occurrence is added to the 'priors' array
                m.priors(end+1) = size(examples_from_this_class, 1) / size(train_labels,1);
			
            end

        end

        % the more training examples the longer training phase...
        % classification phase time is independent of number of training examples...
        function predictions = predict(m, test_examples)

            predictions = categorical; % empty categorical array to store predicitons

            % loops through all examples using the height of the test examples data
            for i=1:size(test_examples,1)

				% prints the number of the current test example being classified
                fprintf('classifying example %i/%i\n', i, size(test_examples, 1));
                % extracts one example from test examples data
                this_test_example = test_examples{i,:};
                % each test example is passed through the predict_one function
                this_prediction = mynb.predict_one(m, this_test_example);
                % adds the prediction to the end of the 'predictions' categorical array 
                predictions(end+1) = this_prediction;
			end
        end

        function prediction = predict_one(m, this_test_example)

            % loops through all of the unique training labels
            for i=1:m.n_classes

                % calculates the extracted test example 'Normal Distribution' based on the current label
                this_likelihood = mynb.calculate_likelihood(m, this_test_example, i);
                % gets the name of the current label/class
                this_prior = mynb.get_prior(m, i);
                % calculates the probability of the current label/class
                % using the likelihood & prior for the current label/class
                posterior_(i) = this_likelihood * this_prior;
            end
            
            % shows the largest value in the 'posterior_' array and its index
            [winning_value_, winning_index] = max(posterior_);
            % predicted label = the index of the max value in the 'posterior_' array 
            % within the unique labels/classes array
            prediction = m.unique_classes(winning_index);

        end
        
        function likelihood = calculate_likelihood(m, this_test_example, class)
            
			likelihood = 1; % sets the default likelihood to 1
            % loops through the extracted test example's columns
			for i=1:length(this_test_example)
                % calculates 'Normal Distribution' of each column within the test example against the current label
                likelihood = likelihood * mynb.calculate_pd(this_test_example(i), m.means{class}(i), m.stds{class}(i));
            end
            
        end
        
        function prior = get_prior(m, class)
            
            % returns the occurrence of the specified class within the
            % training examples as a decimal/fraction
			prior = m.priors(class);
        
        end
        
        % calculates the 'Probability Density' / 'Normal Distributions' (never 0)
        function pd = calculate_pd(x, mu, sigma)
			first_bit = 1 / sqrt(2*pi*sigma^2);
            second_bit = - ( ((x-mu)^2) / (2*sigma^2) );
            pd = first_bit * exp(second_bit);
            
		end
            
    end
    
end