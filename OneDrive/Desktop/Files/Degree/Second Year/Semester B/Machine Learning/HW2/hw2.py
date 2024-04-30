import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    S_size = data.shape[0] # Number of total instanses in data
    
    labels_count = np.unique(data[:, -1], return_counts=True)[1] # Gets how much instanses there is with specific label
    sigma = np.sum((labels_count / S_size) ** 2) # Summing the squar of probabilties

    gini = 1 - sigma
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    S_size = data.shape[0] # Number of total instanses in data
    labels_count = np.unique(data[:, -1], return_counts=True)[1] # Gets how much instanses there is with specific label

    sigma = np.sum((labels_count / S_size) * np.log2(labels_count / S_size)) # Calculate impurty of edibles

    entropy = -sigma
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        S_size = self.data.shape[0] # Number of total instanses in data

        edibles = self.data[self.data[:,-1] == 'e'] # all the edibles rows
        edible_size = edibles.shape[0] # amout of edibles

        poisonous_size = S_size - edible_size # amout of poisonous
        
        if poisonous_size <= edible_size:
            pred = "edible"
        else:
            pred = "poisonous"

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.feature == -1:
            self.feature_importance = 0
            return
        
        sigma = 0

        values = np.unique(self.data.T[self.feature]) # Gets an array of all the values that possible to gets from feature
            
        for value in values:
            new_data = self.data[self.data[:, self.feature] == value] # all the rows that corresponding to the value of the feature
            value_size = new_data.shape[0] # amout of rows that corresponding to the value of the feature
            sigma += (value_size / n_total_sample) * self.impurity_func(new_data)

        S_size = self.data.shape[0] # Number of total instanses in data
        impurty_S = self.impurity_func(self.data)
        self.feature_importance = (S_size / n_total_sample) * impurty_S - sigma

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        sigma = 0
        sigma_info = 0 # Only for GainRatio

        S_size = self.data.shape[0] # Number of total instanses in data

        values = np.unique(self.data.T[feature]) # Gets an array of all the values that possible to gets from feature
            
        for value in values:
            groups[value] = self.data[self.data[:, feature] == value] # all the rows that corresponding to the value of the feature
            value_size = groups[value].shape[0] # amout of rows that corresponding to the value of the feature

            if not self.gain_ratio: # goodness of split
                sigma += (value_size / S_size) * self.impurity_func(groups[value])
            else: # gain ratio
                sigma += (value_size / S_size) * calc_entropy(groups[value])
                sigma_info += (value_size / S_size) * np.log2(value_size / S_size)
        
        if not self.gain_ratio: # goodness of split
            impurty_S = self.impurity_func(self.data)
            goodness = impurty_S - sigma
        else: # gain ratio
            entropy_S = calc_entropy(self.data)
            info_gain = entropy_S - sigma
            split_info = -sigma_info
            
            if split_info == 0: # In situation where we need to dived by zero
                return 0, groups

            goodness = info_gain / split_info # GainRation
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        features = self.data.shape[1] - 1 # Gets an array of all the values that possible to gets from feature
        max_feature_goodness = 0
        max_feature_groups = {}

        # Checking if it possible to create more children
        if self.depth == self.max_depth or self.terminal:
            self.terminal = True
            return

        # Finding the best feature
        for feature in range(features):
            feature_goodness, feature_groups = self.goodness_of_split(feature)
            if max_feature_goodness < feature_goodness:
                max_feature_goodness = feature_goodness
                max_feature_groups = feature_groups
                self.feature = feature

        # Checking if there is helpful feature
        if max_feature_goodness == 0:
            self.terminal = True
            return
        
        values = np.unique(self.data.T[self.feature]) # Gets an array of all the values that possible to gets from feature

        # Dealing with chi test
        if not check_chi(self.chi, self.data, values, max_feature_groups): 
            self.terminal = True
            return
            
        # Creating the children nodes
        for value in values:
            children_node = DecisionNode(max_feature_groups[value], self.impurity_func, -1, self.depth + 1, self.chi, self.max_depth, self.gain_ratio)
            self.add_child(children_node, value)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def check_chi(chi_value, data, values, max_feature_groups):
    """
    This function checks if the chi spliting is randomaize or is it 
    has a probabilty differences
    """
    if chi_value != 1: 
            labels, counts = np.unique(data.T[-1], return_counts=True)

            counts = counts / data.shape[0]
            chi_sqaure = 0
            
            # print(f"values: {values}")
            # print(f"max_feature_groups: {max_feature_groups}")

            # Calculating X^2 
            for value in values: 
                # print(f"value: {value}")
                Df = len(max_feature_groups[value])   
                for label in range(len(labels)):
                    expected = counts[label] * Df
                    observed = len(max_feature_groups[value][max_feature_groups[value][:,-1] == labels[label]])
                    chi_sqaure += ((observed - expected) ** 2) / expected

            degree_of_freedom = len(values) - 1

            # Checking if the distrebution is randomize or have predictive power
            if chi_sqaure < chi_table[degree_of_freedom][chi_value]:
                return False
    return True

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(self.data, self.impurity_func, chi = self.chi,max_depth = self.max_depth, gain_ratio = self.gain_ratio) # Creating the root node
        
        nodes_queue = [] # Holding the nodes that supposed to split
        nodes_queue.append(self.root)

        while len(nodes_queue) > 0: # Loop over the nodes in the tree
            current_node = nodes_queue.pop(0) 
            current_node.calc_feature_importance(current_node.data.shape[0])
            current_node.split() # Create new children nodes
            nodes_queue += current_node.children

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        
        while not node.terminal:
            try:
                value = instance[node.feature] # Gets the value of the feature
                value_index = node.children_values.index(value) # Gets the index of the value in childern values attribute
                node = node.children[value_index]
            except:
                break

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        num_of_correctness = 0
        for row in dataset: # Counting how many correct predictions there are
            prediction = self.predict(row)
            if prediction[0] == row[-1]:
                num_of_correctness += 1
        
        accuracy = num_of_correctness * 100 / dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        train_tree = DecisionTree(data = X_train, impurity_func = calc_entropy, gain_ratio = True, max_depth = max_depth) # Creating the training tree

        train_tree.build_tree() # Building the training tree

        training.append(train_tree.calc_accuracy(X_train))
        validation.append(train_tree.calc_accuracy(X_validation))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation

def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for p_value in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        train_tree = DecisionTree(data = X_train, impurity_func = calc_entropy, gain_ratio = True, chi = p_value) # Creating the training tree

        train_tree.build_tree() # Building the training tree

        chi_training_acc.append(train_tree.calc_accuracy(X_train))
        chi_testing_acc.append(train_tree.calc_accuracy(X_test))
        depth.append(tree_depth(train_tree.root))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth

def tree_depth(root):
    """ Calculates the trees at the given root depth.

    Input:
    - root: the root, DecisionNode, of the DecisionTree.

    Output:
    - depth: the depth of the tree.
    """

    if root.terminal:
        return root.depth
    
    depth_list = [tree_depth(node) for node in root.children]
    return np.max(depth_list)

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node.terminal: # Base case 
        return 1
    
    n_nodes = np.sum([count_nodes(child) for child in node.children]) + 1 # Recursion
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






