from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Transpose the sparse matrix to use questions as the observations
    question_matrix = matrix.T

    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(question_matrix)

    acc = sparse_matrix_evaluate(valid_data, mat.T)

    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = (1, 6, 11, 16, 21, 26)

    # User-based collaborative filtering
    valid_acc_set = []

    # Compute validation accuracy for each k
    for i in range(len(k_values)):
        k = k_values[i]
        acc_k = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_acc_set.append(acc_k)

    # Create plot
    plt.plot(k_values, valid_acc_set, color="navy")
    plt.xlabel("k - Number of Nearest Neighbours")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.show()

    # k=11 has the best validation accuracy
    k_star = 11

    # Find test accuracy for k*=11
    nbrs = KNNImputer(n_neighbors=k_star)
    mat = nbrs.fit_transform(sparse_matrix)
    acc_k_star = sparse_matrix_evaluate(test_data, mat)
    print("Test Accuracy with k*: {}".format(acc_k_star))

    # Item-based collaborative filtering
    valid_acc_set1 = []

    # Compute validation accuracy for each k
    for i in range(len(k_values)):
        k = k_values[i]
        acc_k = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc_set1.append(acc_k)

    # Create plot
    plt.plot(k_values, valid_acc_set1, color="orange")
    plt.xlabel("k - Number of Nearest Neighbours")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.show()

    # k=21 has the best validation accuracy
    k_star1 = 21

    # Find test accuracy for k*=21
    nbrs = KNNImputer(n_neighbors=k_star1)
    mat = nbrs.fit_transform(sparse_matrix.T)
    acc_k_star = sparse_matrix_evaluate(test_data, mat.T)

    print("Test Accuracy with k*: {}".format(acc_k_star))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

