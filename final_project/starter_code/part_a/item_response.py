# TODO figure out a better way to
if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils import *
else:
    from ..utils import *


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # TODO - add clarifying comment
    lklihoods = likelihoods(data, theta, beta)

    # These are the targets
    targets = np.array(data['is_correct'])

    # Compute log likelihood for each (i,j) pair
    log_lklihoods = np.multiply(targets, np.log(
        lklihoods)) + np.multiply(1-targets, np.log(1-lklihoods))
    # need to test out numeric stability
    # log function may freak if exp(theta_i - beta_j) is close to 0
    #TODO - make numerically stable
    """
    thetas = np.array(theta[data['user_id']])
    betas = np.array(beta[data['question_id']])
    targets = np.array(data['is_correct'])

    log_denom = np.log(1 + np.exp(thetas - betas))
    log_lklihoods = targets * (thetas - betas - log_denom) - (1 - targets) * log_denom
    """
    log_lklihood = log_lklihoods.sum()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    # Update theta
    lklihoods = likelihoods(data, theta, beta)
    # Subtract lklihoods from targets
    partials = np.array(data['is_correct']) - lklihoods
    # Group by i (user_id) and sum
    theta = theta + lr * np.bincount(data['user_id'], weights=partials)

    # are more steps needed here?

    # Update beta using new theta?
    lklihoods = likelihoods(data, theta, beta)
    # Subtract targets from lklihoods
    partials = lklihoods - np.array(data['is_correct'])
    # Group by j (question_id) and sum
    beta = beta + lr * np.bincount(data['question_id'], weights=partials)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta. 
    # TODO -- (see if this is the best initialization!)
    theta = np.repeat(np.array([10]), len(np.unique(data['user_id'])))
    beta = np.repeat(np.array([0]), len(np.unique(data['question_id'])))

    performance = {
        'train_NLLK': [],
        'val_NLLK': [],
        'val_acc_lst': []
    }

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        performance['train_NLLK'].append(neg_lld) #
        performance['val_NLLK'].append(neg_log_likelihood(val_data, theta=theta, beta=beta)) #

        score = evaluate(data=val_data, theta=theta, beta=beta)
        performance['val_acc_lst'].append(score) #
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, performance #val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    iterations = [50, 100, 250, 500, 1000]

    final_thetas = []
    final_betas = []
    final_val_acc = []
    for lr in lrs:
        for num_iterations in iterations:
            print(f'Testing model with {lr} learning rate and {num_iterations} iterations...')
            theta, beta, performance = irt(train_data, val_data, lr, num_iterations)
            final_thetas.append(theta)
            final_betas.append(beta)
            final_val_acc.append(performance['val_acc_lst'][-1])

    # Plot heatmap of accuracy
    g = sns.heatmap(np.array(final_val_acc).reshape(5,5))
    g.set_xticklabels(iterations)
    g.set_yticklabels(lrs)
    plt.savefig('irt_gridsearch.png')
    plt.show()

    # Choose optimal hyperparameters
    best_lr = lrs[np.argmin(final_val_acc) // 5]
    best_iter = iterations[np.argmin(final_val_acc) % 5]
    print(f'Optimal hyperparameters: {best_lr} learning rate with {best_iter} iterations')

    # Report NLLK vs. iteration for both training and validation sets
    theta, beta, performance = irt(train_data, val_data, best_lr, best_iter)
    plt.plot(np.arange(best_iter), -np.array(performance['train_NLLK']), color='blue', label='Training LLK')
    plt.plot(np.arange(best_iter), -np.array(performance['val_NLLK']), color='orange', label='Validation LLK')  

    plt.xlabel('Number of Iterations (t)')
    plt.ylabel('Log-Likelihood (LLK)')
    plt.legend(loc='upper right')
    plt.title(
        'Log-Likelihood of Training & Validation\nSets using Item Response Theory Model')

    plt.savefig('llk_graph.png')
    plt.show()

    # Report final validation and test accuracies
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f'Final Validation Accuracy: {val_acc}')
    print(f'Final Test Accuracy: {test_acc}')

    # Five question as a function of theta
    plt.plot(np.arange(len(theta)), sigmoid(theta - beta[0]), color='red', label='j1')
    plt.plot(np.arange(len(theta)), sigmoid(theta - beta[1]), color='orange', label='j2')
    plt.plot(np.arange(len(theta)), sigmoid(theta - beta[2]), color='green', label='j3')
    plt.plot(np.arange(len(theta)), sigmoid(theta - beta[3]), color='blue', label='j4')
    plt.plot(np.arange(len(theta)), sigmoid(theta - beta[4]), color='purple', label='j5')

    plt.xlabel('Theta (θ_i)')
    plt.ylabel('Probability of Correct Response (p(c_ij = 1) | θ, β )')
    plt.legend(loc='upper right')
    plt.title(
        'Probability Each Student Has of Correctly Answering 5 Questions')

    plt.savefig('5q_prob_plot.png')
    plt.show()
    

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
