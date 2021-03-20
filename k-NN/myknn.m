classdef myknn
    
    methods(Static)
        
        % no parameters and one hyperparameter (k)
        function m = fit(train_examples, train_labels, k)
            
            % start of standardisation process (z-score standardisation)
            % implementing standardsation to re-scale all of the values 
            % to avoid features wihtin the table dominating the Euclidean distance calculations
			m.mean = mean(train_examples{:, :}); % mean of all training examples
			m.std = std(train_examples{:, :}); % standard deviation of all training examples
            
            % loops through all examples using the height of the train examples column
            for i = 1:1:size(train_examples, 1)
                % standardises each training exmaple...
				train_examples{i, :} = train_examples{i, :} - m.mean;
                train_examples{i, :} = train_examples{i, :} ./ m.std;
            end
            % end of standardisation process
            
            % after standardisation, training examples & labels are returned
            m.train_examples = train_examples;
            m.train_labels = train_labels;
            
            % no. of neighbours considered when classifying an example
            m.k = k; % entered when calling the fit() method within the 3rd argument
        
        end

        function predictions = predict(m, test_examples)

            predictions = categorical; % empty categorical array to store predicitons
            
            % loops through all examples using the height of the test examples
            for i=1:size(test_examples, 1)
                
                % prints the number of the current test example being classified
                fprintf('classifying example example %i/%i\n', i, size(test_examples, 1));
                
                % extracts one example from test examples column
                this_test_example = test_examples{i,:};
                
                % start of standardisation process
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                % each test example is passed through the predict_one function
                this_prediction = myknn.predict_one(m, this_test_example);
                
                % adds the prediction to the end of the 'predictions' categorical array 
                predictions(end+1) = this_prediction;
            end
            
        end

        % slower classification the more training examples there are...
        function prediction = predict_one(m, this_test_example)
            
            % calulcates the straight line distance between two co-oridinates
            % (the testing example and all the training examples)
            % returns the distances array containing all Euclidean distances
            distances = myknn.calculate_distances(m, this_test_example);
            
            % calculates the k shortest distances between the test example
            % and all the training examples, e.g. k = 10 (10 nearest neighbours)
            % returns the neighbour_indices array containing the row number
            % of the training example's data which are the k nearest neighbours
            neighbour_indices = myknn.find_nn_indices(m, distances);
            
            % predict the class of the current testing example based the most common class
            % of the current testing example's k nearest neighbours
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end

        function distances = calculate_distances(m, this_test_example)
            
			distances = []; % empty array
            % loops through all examples using the height of the train examples
            for i=1:size(m.train_examples, 1)
                
                % extracts one training example from train examples
				this_training_example = m.train_examples{i, :};
                
                % calculates the distance between the current testing & training examples
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                
                % adds the calculated Euclidean distance to the end of the array
                distances(end + 1) = this_distance;
            end
            
        end

        function distance = calculate_distance(p, q)
            % calculating the Euclidean distance between any 2 examples
			differences = q - p; % calculates the difference between training example's data & testing example's data 
            squares = differences .^ 2; % squares each value within the row (converts to positive value)
            total = sum(squares); %  sums all the values within the row
            distance = sqrt(total); % square root of the total to calculate the Euclidean distance 
        
        end

        % low k = overfitting training data --- % high k = underfitting training data 
        function neighbour_indices = find_nn_indices(m, distances)
            
            % sorts the distances array & stores the indices of the sorted array
			[sorted, indices] = sort(distances);
            % extracts the k shortest distances, e.g. 1 to 10, if k = 10)
            neighbour_indices = indices(1:m.k);
        
		end
        
        function prediction = make_prediction(m, neighbour_indices)

            % extracts the corresponding labels for the neighbour indices array 
            % (the neighbour indices array is based on the current testing example)
			neighbour_labels = m.train_labels(neighbour_indices);
            
            % returns the most common label as a prediction
            prediction = mode(neighbour_labels);
        
		end

    end
    
end

