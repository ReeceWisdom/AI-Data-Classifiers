classdef mytree
    
    methods(Static)
        
        function m = fit(train_examples, train_labels)
            
			emptyNode.number = []; % stores the unique number of nodes
            emptyNode.examples = []; % stores training examples
            emptyNode.labels = []; % stores training labels
            emptyNode.prediction = []; % a prediction basd on any class labels the node holds
            emptyNode.impurityMeasure = []; % numeric measurement of the impurity of any class labels held by a node
            emptyNode.children = {}; % stores the split node's data
            emptyNode.splitFeature = []; % stores the column number which defines the split
            emptyNode.splitFeatureName = []; % stores the name which defines the split
            emptyNode.splitValue = []; % stores the value which defines the split

            m.emptyNode = emptyNode;
            
            r = emptyNode; % creates an root empty node
            r.number = 1; % assigns 1 as the root node's number
            r.labels = train_labels; % copies all model training labels
            r.examples = train_examples; % copies all model training examples
            r.prediction = mode(r.labels); 
            % generates single class prediction for the data 
            % IF the root node cannot be further split

            m.min_parent_size = 10; % minimum number of examples a node must contain before considering splitting
            % (allowing deep decision trees can lead to overfitting training data)
            m.unique_classes = unique(r.labels); % list of all the unique training labels
            m.feature_names = train_examples.Properties.VariableNames; % list of all the examples' individual features
			m.nodes = 1; % current number of nodes in the tree           (copying the names of the table's column headers)
            m.N = size(train_examples, 1); % total number of training examples used to train the model

            m.tree = mytree.trySplit(m, r);
            % generates the tree by passing the root node to the function. 
            % the function tests to see whether a node can be split into two child nodes with a reduced overall impurity 
            % the function is recusive and only returns once it's no longer possible to split any more nodes 
            % either due because the nodes don't contain enough training examples OR 
            % because no split is available that will reduce the overall impurity

        end
        
        function node = trySplit(m, node)
            
            % checks whether the current node is large enough to split
            if size(node.examples, 1) < m.min_parent_size
				return
            end
            
            % measures the current impurity of the node's class labels using Gini's Diversity Index (GDI)
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);

            % loops through all the current node's data features (4)
            for i=1:size(node.examples, 2)
                
                % evaulates all the possible splits on the data's features (column names)
				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples, 2));
                % sorts & indexes the examples based on the current feature
				[ps,n] = sortrows(node.examples, i);
                % stores the labels of the sorted examples based on index
                ls = node.labels(n);
                
                biggest_reduction(i) = -Inf; 
                biggest_reduction_index(i) = -1;
                biggest_reduction_value(i) = NaN;
                
                % considers splitting the current node's stored training data 
                % on every unique value (inner loop) of every feature (outer loop)
                % if all features' values are unique, then 
                % the number of possible splits = no. of examples - 1
                for j=1:(size(ps,1)-1)
                    % if the next feature value is the same as the current value it's skipped
                    if ps{j,i} == ps{j+1,i}
                    % (prevents trying to split on the same value more than once)
                        continue;
                    end
                    
                    % after generating all the possible splits from every unique value of every feature
                    % a split is chosen based on which split reduces the impurity amongst the labels the most (GDI)
                    
                    % Then, the GDI of both potential child nodes are calulcated, added and subtracted from the parent node,
                    % if the result is positive then the split produces a reduction in impurity
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    
                    % if the current split's reduction in impurity is greater than 
                    % the previous reduction, the current becomes the biggest reduction
                    % and is stored in the 'biggest_reduction' array
                    if this_reduction > biggest_reduction(i)
                        biggest_reduction(i) = this_reduction;
                        % stores the index of each biggest reduction
                        biggest_reduction_index(i) = j;
                    end
                end
				
            end
            
            
            % winning reduction = value of greatest reduction within the array
            % winning feature = feature of which the winning reduction value
            [winning_reduction,winning_feature] = max(biggest_reduction);
            % winning index = poisition of the value within the feature's column 
            winning_index = biggest_reduction_index(winning_feature);

            % if the node's impurity is greater than 0, it did not produce 
            % a reduction in impurity and is returned
            if winning_reduction <= 0
                return
            % if the node's impurity is less than or equal to 0,
            % it did produce a reduction in impurity and is split...
            else
                % sorts & indexes the examples based on the feature of which the split happen 
                [ps,n] = sortrows(node.examples,winning_feature);
                % stores the labels of the examples based on their index
                ls = node.labels(n);

                % stores the index of the table column (label) of which the split will occur 
                node.splitFeature = winning_feature;
                % stores the name of the table column (label) of which the split will occur
                node.splitFeatureName = m.feature_names{winning_feature};
                % stores the value for which the split occur
                % this method of calculation is used to determine the exact value 
                % between the chosen value and the next value
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;

                % deletes current node's examples & labels as they will
                % move to the 2 child nodes after splitting
                node.examples = [];
                node.labels = []; 
                % deletes current node's predicition as it will never be
                % used during the prediction phase
                node.prediction = [];

                node.children{1} = m.emptyNode; % creates child node 1
                m.nodes = m.nodes + 1; % adds 1 to the total number of unique nodes
                node.children{1}.number = m.nodes; % assigns the new unique number to the new child node
                % populates the node with examples from the 1st example to the index of the chosen split value
                node.children{1}.examples = ps(1:winning_index,:);
                % populates the node with labels from the 1st label to the index of the chosen split value
                node.children{1}.labels = ls(1:winning_index);
                % a prediciton based on the most common label within the new child node
                node.children{1}.prediction = mode(node.children{1}.labels);
                
                node.children{2} = m.emptyNode; % creates child node 2
                m.nodes = m.nodes + 1; % adds 1 to the total number of unique nodes
                node.children{2}.number = m.nodes; % assigns the new unique number to the new child node
                % populates the node with examples from the index of the chosen split value to the end
                node.children{2}.examples = ps((winning_index+1):end,:); 
                % populates the node with labels from the chosen split value to the end
                node.children{2}.labels = ls((winning_index+1):end);
                % a prediciton based on the most common label within the new child node
                node.children{2}.prediction = mode(node.children{2}.labels);
                
                % performs the trySplit function on both new child nodes
                % continues recursively until a split is no longer possible
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        
        function e = weightedImpurity(m, labels)
            
            % weight variable is used re-scale the GDI scores 
            % so a fair comparison is made between the impurity of a parent node vs 2 potential child nodes
            weight = length(labels) / m.N;

            summ = 0;
            obsInThisNode = length(labels);
            for i=1:length(m.unique_classes)
                
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                summ = summ + (pc*pc);
            
			end
            g = 1 - summ;
            e = weight * g;

        end

        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            % loops through all test examples
            for i=1:size(test_examples,1)
                
				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                % extracts the current test example
                this_test_example = test_examples{i,:};
                % passes the current test example to 'predict_one' function
                this_prediction = mytree.predict_one(m, this_test_example);
                % adds the prediction for the current example to a 'predicitions' array
                predictions(end+1) = this_prediction;
            
			end
        end

        function prediction = predict_one(m, this_test_example)
            
            % calls the 'decend_tree' function to make a prediction
			node = mytree.descend_tree(m.tree, this_test_example);
            % each test example's corresponding leaf node prediction is returned
            % to the predicition field (most common)
            prediction = node.prediction;
        
		end
        
        function node = descend_tree(node, this_test_example)
            
            % if the current node is a leaf node (no child node) = return
			if isempty(node.children)
                return;
            else
                % decends the tree via comparing the current test example 
                % against each node's split feature & split value
                % once a leaf node is reached, that node's class predicition is returned
                if this_test_example(node.splitFeature) < node.splitValue
                    % recursive function calling itself until a leaf node is found
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                else
                    % recursive function calling itself until a leaf node is found
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
            % if a leaf node has been reached...
			if isempty(node.children)
                % print the leaf node's number and class predicition
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                % print the node's number and it's split rule
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                % decend the tree...
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end