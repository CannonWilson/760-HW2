import os
import math
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 

"""
Note: change c_val comparison to >0
"""

def load_data(path):
    """
    Given a path, returns a lists of dictionaries
    with the following fields/values:
    {
        x_1: [0-1],
        x_2: [0-1],
        y: {0, 1}
    }
    """
    if os.path.exists(path) and path[-4:] == ".txt":
        data_list = []
        with open(path) as f:
            file_text = f.read()
        file_lines = file_text.split('\n')
        for line_idx in range(len(file_lines)):
            split_line = file_lines[line_idx].split(" ")
            if len(split_line) == 3:
                file_lines[line_idx] = {
                    "x_1": float(split_line[0]),
                    "x_2": float(split_line[1]),
                    "y": int(split_line[2])
                }
        return file_lines
    else:
        raise Exception("That is not a valid \
            path to a data text file")

def DetermineCandidateNumericSplits(training_instances_d):
    """
    Returns a list of tuples where the first item is the float
    representing the split value and the second item is a string
    representing the feature name.
    """
    candidate_splits = [] # initialize set of candidate split
    # let v_j denote the value of Xi for the jth data point
    # sort the dataset using vj as the key for each data point
    for feature_x_i in [1,2]:
        sorted_dataset = sorted(training_instances_d, \
            key=lambda instance_dict: instance_dict[f"x_{feature_x_i}"]) # sort instances by values of one feature
        # print('sorted_dataset: ', sorted_dataset)
        for idx in range(len(sorted_dataset) - 1):     # for each pair of adjacent vj, vj+1 in the sorted order
            if sorted_dataset[idx]["y"] != sorted_dataset[idx+1]["y"]:    # if the corresponding class labels are different
                # add candidate split Xi ≤ (vj + vj+1 )/2 to C
                # candidate_val_1 = sorted_dataset[idx][f"x_{feature_x_i}"]
                candidate_val_2 = sorted_dataset[idx+1][f"x_{feature_x_i}"]
                # candidate_splits.append((candidate_val_1, f"x_{feature_x_i}"))
                candidate_splits.append((candidate_val_2, f"x_{feature_x_i}"))
    # print('returning splits: ', candidate_splits)
    return candidate_splits

def is_stopping_criteria_met(training_instances_d, \
    candidate_splits, gain_ratios, entropy_list):
    # Calculate the majority class
    zero_count = 0
    one_count = 0
    majority_class = 1 # predict 1 when no majority class
    for instance in training_instances_d:
        if instance['y'] == 0: zero_count+=1
        elif instance['y'] == 1: one_count+=1
    if zero_count > one_count:
        majority_class = 0

    # Check if node is empty or there are no splits
    if len(training_instances_d) <= 1 or len(candidate_splits) == 0:
        # print('stopping on empty')
        return (True, majority_class)

    # Check if any split has zero entropy
    # if 0 in entropy_list:
    #     print("stopping on zero entropy")
    #     return (True, majority_class)
    
    # If not, check if every split has a zero gain ratio
    for ratio in gain_ratios:
        if ratio != 0:
            return (False, None)
    
    # If we make it here, every split has a zero gain ratio. Stop!
    # print("Stopping on zero gain ratio")
    return (True, majority_class)


def FindBestSplit(training_instances_d, candidate_splits, gain_ratios):
    """
    Given some data (a list of dictionaries as shown
    in load_data), and candidate splits (a list of 
    tuples like (c, f) where c is split value and
    f is the name of the feature) return the split
    that maximizes the gain ratio. Returns one 
    tuple like (c, f).

    Entropy formula:
    H = sum over c (-p_c * log_2(p_c))
    """
    
    # Find the best split (highest gain ratio)
    best_split_idx = np.argmax(gain_ratios)
    best_split = candidate_splits[best_split_idx]
    # if force_split:
    #     best_split_idx = 0
    #     best_split = (training_instances_d[0]['x_1'], 'x_1')
    best_split_val = best_split[0]
    best_split_ftr = best_split[1]

    # Split the training instances and return two lists
    under_split_instances = []
    over_split_instances = []
    for instance in training_instances_d:
        if instance[best_split_ftr] >= best_split_val:
            over_split_instances.append(instance)
        else:
            under_split_instances.append(instance)

    return (under_split_instances, over_split_instances, best_split)

first_time = True
def calc_gain_ratio_and_entropy(training_instances_d, candidate_splits):

    if len(training_instances_d) == 0 or len(candidate_splits) == 0:
        return ([], [])

    global first_time
    # Calculate the entropy of the training instances
    num_class_0 = 0
    num_class_1 = 0
    for instance in training_instances_d:
        if instance['y'] == 0:
            num_class_0 += 1
        elif instance['y'] == 1:
            num_class_1 += 1
    total = num_class_0 + num_class_1
    instance_prob_0 = num_class_0 / total
    instance_prob_1 = num_class_1 / total
    instances_entropy = 0 if (instance_prob_0 == 0 or instance_prob_1 == 0) \
        else -1*(instance_prob_0 * math.log2(instance_prob_0)) \
        - instance_prob_1 * math.log2(instance_prob_1)

    # Calculate info gain ratio for each split
    split_gain_ratios = []
    split_entropy_list = []
    for split in candidate_splits:
        class_0_under_split = 0
        class_1_under_split = 0
        class_0_over_split = 0
        class_1_over_split = 0
        for instance in training_instances_d: # loop over training instances
            c_val = split[0]
            ftr_name = split[1]
            if instance[ftr_name] >= c_val:
                if instance['y'] == 0:
                    class_0_over_split += 1
                elif instance['y'] == 1:
                    class_1_over_split += 1
            else:
                if instance['y'] == 0:
                    class_0_under_split += 1
                elif instance['y'] == 1:
                    class_1_under_split += 1
            
        total_under = class_0_under_split + class_1_under_split
        total_over = class_0_over_split + class_1_over_split
        cond_entr_given_under = 0 if (class_0_under_split == 0 or \
            class_1_under_split == 0) else \
            -1 * (class_0_under_split/total_under) * \
            math.log2(class_0_under_split/total_under) - \
                (class_1_under_split/total_under) * \
                    math.log2(class_1_under_split/total_under)
        cond_entr_given_over = 0 if (class_0_over_split == 0 or \
            class_1_over_split == 0) else \
            -1 * (class_0_over_split/total_over) * \
            math.log2(class_0_over_split/total_over) - \
                (class_1_over_split/total_over) * \
                    math.log2(class_1_over_split/total_over)
        under_total = class_0_under_split + class_1_under_split
        over_total = class_0_over_split + class_1_over_split
        instances_total = len(training_instances_d)
        weight_under = under_total / instances_total
        weight_over = over_total / instances_total
        cond_entr_given_split = weight_under * cond_entr_given_under + \
            weight_over * cond_entr_given_over
        split_info_gain = instances_entropy - cond_entr_given_split
        split_entropy = 0 if (weight_under == 0 or weight_over == 0) else \
            -1 * weight_under * math.log2(weight_under) - \
            weight_over * math.log2(weight_over)
        if split_entropy == 0: split_gain_ratio = 0
        else: split_gain_ratio = split_info_gain / split_entropy
        split_gain_ratios.append(split_gain_ratio)
        split_entropy_list.append(split_entropy)
        # if first_time:
        #     if split_entropy == 0:
        #         print(f'For split {split}, mutual info: {split_info_gain}')
        #     else:
        #         print(f'For split {split}, info gain ratio: {split_gain_ratio}')
    first_time = False
    return (split_gain_ratios, split_entropy_list)

tree = []

def MakeSubtree(training_instances_d, depth_idx, width_idx):
    """
    Create a subtree! Starts with tree as empty list.

    At each new depth level (including the root), a 
    new list is appended to the tree list. These 
    nested lists will contain the nodes, which are merely
    dictionaries with two fields/values: feature_dim,
    (values in range 0,1) and threshold (float values).
    Candidate splits choose the left sub-branch if 
    x_j >= threshold or the right sub-branch otherwise.

    Each item in the tree will be a split--of type 
    tuple like (c, f) where c is the split value 
    and f is the name of the feature --or an int y,
    the class decision, if the item represents a leaf node.
    """
    # print('training instances: ', training_instances_d)
    if depth_idx == len(tree): # new layer needed
        num_nodes = 2**depth_idx # num nodes at depth d is 2^d
        tree.append([])
        tree[depth_idx] = [None for _ in range(num_nodes)]

    candidate_splits = DetermineCandidateNumericSplits(training_instances_d)
    gain_ratios, entropy_list = \
        calc_gain_ratio_and_entropy(training_instances_d, candidate_splits)
    do_stop, class_decision = \
        is_stopping_criteria_met(training_instances_d, \
            candidate_splits, gain_ratios, entropy_list)
    if do_stop:
        tree[depth_idx][width_idx] = class_decision # make a leaf node
    else:
        under_split_instances, over_split_instances, best_split = \
            FindBestSplit(training_instances_d, candidate_splits, gain_ratios)
        # make an internal node N
        tree[depth_idx][width_idx] = best_split
        left_child_w_idx = 2 * width_idx
        right_child_w_idx = 2 * width_idx + 1 
        MakeSubtree(over_split_instances, depth_idx+1, left_child_w_idx)
        MakeSubtree(under_split_instances, depth_idx+1, right_child_w_idx)
    return tree

def visualize_tree(vis_tree):

    x = []
    y = []
    annotations = []

    tree_depth = len(vis_tree)
    tree_max_width = len(vis_tree[-1]) # might need to make this smarter for better scaling

    for depth_i in range(len(vis_tree)):
        distance_count = len(vis_tree[depth_i]) + 1 # also equal to {2}^{d_i} + 1
        equal_distance = tree_max_width / distance_count
        for width_i in range(len(vis_tree[depth_i])):
            x_coord = -0.5 * tree_max_width + \
                (width_i + 1) * equal_distance
            y_coord = -1 * depth_i
            
            ann = vis_tree[depth_i][width_i]
            if type(ann) == tuple:
                ann = (round(1000 * ann[0]) / 1000, ann[1])

            if ann is not None:
                x.append(x_coord)
                y.append(y_coord)
                annotations.append(ann)
    
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(-tree_max_width/2, tree_max_width/2)
    ax.set_ylim(-tree_depth, 0)
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def classify_pt(instance_dict, tree):
    
    depth_i = 0
    width_i = 0
    while True:
        cur_val = tree[depth_i][width_i]
        if cur_val == None:
            raise Exception('Tree traversal failed. \
                Ran into a None value.')
        if type(cur_val) == int:
            return cur_val
        
        # We have not found the final value. keep going!
        if type(cur_val) == tuple:
            depth_i += 1
            split_val = cur_val[0]
            ftr_name = cur_val[1]
            inst_val = instance_dict[ftr_name]
            if inst_val >= split_val:
                width_i = width_i * 2 # go down left child path
            else:
                width_i = width_i * 2 + 1 # go down right child path
        else:
            raise Exception('I dunno how you got here bud, \
                but you dun fucked up real bad.')


def show_decision_plot(data_pts, vis_tree):
    # print('data: ', data)

    # Idea: scatter plot all of the points
    # Add low opacity dots in the background
    # of two different colors (red, blue)
    # to represent the decision bounds
    x_1 = [inst_dict['x_1'] for inst_dict in data_pts]
    x_2 = [inst_dict['x_2'] for inst_dict in data_pts]
    y = [inst_dict['y'] for inst_dict in data_pts]

    fig = plt.figure()
    ax = fig.add_subplot()

    NUM_BOUND_PTS = 100000
    decision_bounds_x_pts = np.random.uniform(min(x_1), max(x_1), NUM_BOUND_PTS)
    decision_bounds_y_pts = np.random.uniform(min(x_2), max(x_2), NUM_BOUND_PTS)
    decision_colors = []
    BLUE = '#3348ff'
    RED = '#ff3333'

    for bound_i in range(len(decision_bounds_x_pts)):
        pt = {
            'x_1': decision_bounds_x_pts[bound_i],
            'x_2': decision_bounds_y_pts[bound_i]
        }
        pt_class = classify_pt(pt, tree)
        if pt_class == 0:
            decision_colors.append(RED)
        else: 
            decision_colors.append(BLUE)
    ax.scatter(decision_bounds_x_pts, decision_bounds_y_pts, \
        c=decision_colors, marker="s", alpha=0.1)
    ax.scatter(x_1, x_2, c=y, marker="o")

    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('d_8192 decision boundary')
    plt.show()
    
def calculate_accuracy():
    total = 0
    correct = 0
    for instance in data:
        # print(instance)
        # print(instance['y'] == classify_pt(instance, tree))
        if instance['y'] == classify_pt(instance, tree):
            correct +=1
        total += 1
    return 100 * (correct/total)

def calculate_test_accuracy(test_data, tree):
    total = 0
    correct = 0
    for instance in test_data:
        # print(instance)
        # print(instance['y'] == classify_pt(instance, tree))
        if instance['y'] == classify_pt(instance, tree):
            correct +=1
        total += 1
    return 100 * (correct/total)

if __name__ == '__main__':
    data = load_data('./data/Dbig.txt')
    accuracies = []
    # MakeSubtree(data, 0, 0)
    random.shuffle(data)
    test_data = data[:8192]
    test_data_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in test_data]
    test_data_y = [inst_dict['y'] for inst_dict in test_data]

    n = [32, 128, 512, 2048, 8192]
    acc = []

    # tree = []
    d_32 = data[:32]
    d_32_train_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in d_32]
    d_32_train_y = [inst_dict['y'] for inst_dict in d_32]
    clf = DecisionTreeClassifier()
    clf = clf.fit(d_32_train_x,d_32_train_y)
    y_pred = clf.predict(test_data_x)
    acc.append(1-metrics.accuracy_score(test_data_y, y_pred))

    # MakeSubtree(d_32, 0, 0)
    # show_decision_plot(d_32, tree)
    # acc_d_32 = calculate_test_accuracy(test_data, tree)
    # accuracies.append(acc_d_32)
    # print(f'32 finished with {acc_d_32} and tree length: {len(tree)}')

    # tree = []
    d_128 = data[:128]
    d_128_train_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in d_128]
    d_128_train_y = [inst_dict['y'] for inst_dict in d_128]
    clf = DecisionTreeClassifier()
    clf = clf.fit(d_128_train_x,d_128_train_y)
    y_pred = clf.predict(test_data_x)
    acc.append(1-metrics.accuracy_score(test_data_y, y_pred))
    # MakeSubtree(d_128, 0, 0)
    # show_decision_plot(d_128, tree)
    # acc_128 = calculate_test_accuracy(test_data, tree)
    # accuracies.append(acc_128)
    # print(f'128 finished with {acc_128} and tree length: {len(tree)}')


    # tree = []
    d_512 = data[:512]
    d_512_train_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in d_512]
    d_512_train_y = [inst_dict['y'] for inst_dict in d_512]
    clf = DecisionTreeClassifier()
    clf = clf.fit(d_512_train_x,d_512_train_y)
    y_pred = clf.predict(test_data_x)
    acc.append(1-metrics.accuracy_score(test_data_y, y_pred))
    # MakeSubtree(d_512, 0, 0)
    # show_decision_plot(d_512, tree)
    # acc_512 = calculate_test_accuracy(test_data, tree)
    # accuracies.append(acc_512)
    # print(f'512 finished with {acc_512} and tree length: {len(tree)}')

    # tree = []
    d_2048 = data[:2048]
    d_2048_train_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in d_2048]
    d_2048_train_y = [inst_dict['y'] for inst_dict in d_2048]
    clf = DecisionTreeClassifier()
    clf = clf.fit(d_2048_train_x, d_2048_train_y)
    y_pred = clf.predict(test_data_x)
    acc.append(1-metrics.accuracy_score(test_data_y, y_pred))
    # MakeSubtree(d_2048, 0, 0)
    # show_decision_plot(d_2048, tree)
    # acc_2048 = calculate_test_accuracy(test_data, tree)
    # accuracies.append(acc_2048)
    # print(f'2048 finished with {acc_2048} and tree length: {len(tree)}')
    

    # tree = []
    d_8192 = data[:8192]
    d_8192_train_x = [[inst_dict['x_1'], inst_dict['x_2']] for inst_dict in d_8192]
    d_8192_train_y = [inst_dict['y'] for inst_dict in d_8192]
    clf = DecisionTreeClassifier()
    clf = clf.fit(d_8192_train_x, d_8192_train_y)
    y_pred = clf.predict(test_data_x)
    acc.append(1-metrics.accuracy_score(test_data_y, y_pred))
    # MakeSubtree(d_8192, 0, 0)
    # show_decision_plot(d_8192, tree)
    # acc_8192 = calculate_test_accuracy(test_data, tree)
    # accuracies.append(acc_8192)
    # print(f'8192 finished with {acc_8192} and tree length: {len(tree)}')

    # print('accuracies: ||||')
    # print(accuracies)

    # Test results from above code:
    """
    32 finished with 88.92822265625 and tree length: 5
    128 finished with 92.5048828125 and tree length: 9
    512 finished with 94.95849609375 and tree length: 14
    2048 finished with 97.4609375 and tree length: 22
    8192 finished with 100.0 and tree length: 31
    """
    # n = [32, 128, 512, 2048, 8192]
    # acc = [1-.8892822265625, 1-.925048828125, 1-.9495849609375, 
    #     1-.974609375, 1-1.000]

    plt.plot(n, acc)
    plt.xlabel('n')
    plt.ylabel('Err_n')
    plt.title('Sklearn decision tree learning curve')
    plt.show()


    print(f"Finished running! Tree with training accuracy {calculate_accuracy()}:")
    print("TREE", tree)
    # visualize_tree(tree)
    # show_decision_plot(data, tree)

