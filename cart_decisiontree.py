import pandas as pd 
import numpy as np

# Load a excel file
def load_excel(filename):
    data = pd.read_excel(filename)
    data = data.drop('id', 1)
    return data

def split_data_train_test(data): #split data into 90% train and 10% test
    train = pd.DataFrame()
    test = pd.DataFrame()
    train_size =int(0.9 * len(data)) 
    train=train.append(data.iloc[0:train_size,:])
    test=test.append(data.iloc[train_size:len(data),:])
    #x_test=test.iloc[:,0:11]
    #y_test=test.iloc[:,11:12]
    return train.values,test.values

def binary_split(index, value, data):
    left=[]
    right=[]
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right

def gini_cost_function(groups, classes):
    size_parent =  sum([len(group) for group in groups])
    cost=0
    for group in groups:
        size= len(group)
        if size == 0: #aviod to divide on Zero (empty group )
            continue
        sum_gini=0
        for class_val in classes:
            p=[row[-1] for row in group].count(class_val)/size
            sum_gini += p*p
        cost += (1- sum_gini)* (size/size_parent)
    return cost

def best_split(data):
    class_values= list(set(row[-1] for row in data)) #[0, 1]
    init_index, init_value, initi_cost, init_groups=999, 999, 999, None
    for index in range(len(data[0])-1):
        for row in data:
            groups=binary_split(index, row[index], data)
            cost=gini_cost_function(groups, class_values)
            #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], cost))
            #print(groups)
            if cost < initi_cost:
                init_index,init_value,init_groups, initi_cost= index, row[index], groups, cost
    return {'index':init_index, 'value':init_value, 'groups':init_groups}

def leaf_node(group):
    outcomes=[row[-1] for row in group]
    return max(outcomes, key=outcomes.count) #Find the class value with maximum occurrences in a list

def child_splits(node, max_depth, min_size, current_depth):
    left, right= node['groups']
    del(node['groups'])
    #print(left)
    if not left or not right: #check if either left or right group of rows is empty
        node['left'] = node['right']= leaf_node(left + right)
        return 
    if current_depth >= max_depth:  #check for max depth
        node['left'] , node['right'] = leaf_node(left) , leaf_node(right)
        #print(left)
        return
    # check min_size for left and right
    if len(left) <= min_size:
        node['left'] = leaf_node(left)
    else:
        node['left'] = best_split(left)
        child_splits(node['left'] , max_depth, min_size, current_depth+1)
    if len(right) <= min_size:
        node['right'] = leaf_node(right)
    else:
        node['right '] = best_split(right)
        child_splits(node['right '] , max_depth, min_size, current_depth+1)

def build_CART_tree(train, max_depth, min_size):
    root= best_split(train)
    child_splits(root, max_depth, min_size, 1)
    return root

if __name__ == "__main__":
    path="cardio.xlsx"
    data=load_excel(path)
    train,test=split_data_train_test(data)
    #groups=binary_split(6, 3, train[0:10,:])
    #print(gini_cost_function(groups, [0,1]))
    #print(best_split(train[0:10,:]))
    #print(child_splits(best_split(train[0:10,:]),1,1,1))
    print(build_CART_tree(train[0:6999,:], 2, 1))