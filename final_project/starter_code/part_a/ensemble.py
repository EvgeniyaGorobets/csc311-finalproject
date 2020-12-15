import os
import sys
if os.path.dirname(__file__):
    # You are running this from a parent directory (starter_code)
    os.chdir('part_a')
sys.path.append(os.path.dirname(os.getcwd()))
from utils import load_train_csv, load_public_test_csv, load_valid_csv
import numpy as np

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def likelihoods(data, theta, beta):
    # Build two vectors, representing theta_i and beta_j for each
    # user-question (i,  j) pair in the dictionary
    thetas = np.array(theta[data['user_id']])
    betas = np.array(beta[data['question_id']])
    # Compute sigmoid of each theta_i, beta_j pair
    likelihoods = sigmoid(thetas - betas)
    return likelihoods

def update_theta_beta(data, lr, theta, beta):
    # Update theta
    lklihoods = likelihoods(data, theta, beta)
    # Subtract lklihoods from targets
    partials = data['is_correct'] - lklihoods
    # Group by i (user_id) and sum
    theta = theta + lr * np.bincount(data['user_id'], weights=partials)
    lklihoods = likelihoods(data, theta, beta)
    # Subtract targets from lklihoods
    partials = lklihoods - data['is_correct']
    # Group by j (question_id) and sum
    beta = beta + lr * np.bincount(data['question_id'], weights=partials)
    return theta, beta


def irt(data, lr, iterations):
    theta = np.repeat(np.array([10]), len(np.unique(data['user_id'])))
    beta = np.repeat(np.array([0]), len(np.unique(data['question_id'])))
    for i in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)
    return theta, beta 

def evaluate(preds,targets):
    #Given the raw prediction probs, round and return accuracy
    rounded = np.array([round(pred) for pred in preds])
    num_right = np.count_nonzero(rounded-np.array(targets) == 0)
    return num_right/len(targets)

def bagged_predict(data, param_list):
    """ Get an average prediction over each
    of the models trained.

    :param data: data dictionary
    :param param_list: list of parameter tuples
    :return: float
    """
    preds = np.zeros(len(data['is_correct']))
    for model in param_list:
        preds += predict(data,model[0],model[1])
    return preds/len(param_list)

def predict(data, theta, beta):
    """Return vector of float prediction probabilities
    for the given data
    :param theta: Vector
    :param beta: Vector
    :return: Vector
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred


def bootstrap(matrix):
    '''
    Generate a bagged dataset from the given
    by randomly sampling with replacement
    '''
    n = len(matrix['user_id'])
    indices = np.random.randint(0,n,n)
    new_dic = {}
    for key in matrix.keys():
        new_dic[key] = np.array(matrix[key])[indices]
    return new_dic

def main():
    np.random.seed(1337)
    #Load data
    train_data = load_train_csv("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    #Set hyperparameters
    best_lr = 0.01
    best_iter = 500
    num_models = 3
    param_list = []
    #Train each model on bagged data
    for i in range(num_models):
        print('Training Model ' + str(i+1))
        bagged = bootstrap(train_data)
        param_list.append(irt(bagged, best_lr, best_iter))
    
    #Get training bagged predictions and accuracy
    train_bagged_preds = bagged_predict(train_data,param_list)
    train_accuracy = evaluate(train_bagged_preds,train_data['is_correct'])

    #Get validation bagged predictions and accuracy
    val_bagged_preds = bagged_predict(val_data,param_list)
    val_accuracy = evaluate(val_bagged_preds,val_data['is_correct'])

    #Get test bagged predictions and accuracy
    test_bagged_preds = bagged_predict(test_data,param_list)
    test_accuracy = evaluate(test_bagged_preds,test_data['is_correct'])

    print(f'Final Training Accuracy: {train_accuracy}')
    print(f'Final Validation Accuracy: {val_accuracy}')
    print(f'Final Test Accuracy: {test_accuracy}')
    
if __name__ == "__main__":
    main()
