import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

final_results = []

class SoftsvmpolyModel:
    def __init__(self, opt_alphas, opt_k):
        self.opt_alphas = opt_alphas
        self.opt_k = opt_k

    def evaluate_model(self, x_train, x_test, y_test):
        return Softsvmpoly.evaluate_model(
            x_train=x_train,
            alphas=self.opt_alphas,
            k=self.opt_k,
            x_val=x_test,
            y_val=y_test,
        )

    def predict(self, x, x_train):
        return Softsvmpoly.predict(x=x, x_train=x_train, alphas=self.opt_alphas, k=self.opt_k)


class Softsvmpoly:
    @staticmethod
    def train(alpha, train_x: np.array, train_y: np.array):
        """
        a wrapper function to softsvmpoly
        :param alpha: dict that represent a vector of alpha hyperparameters.
        :param train_x:
        :param train_y:

        return Optimal alpha
        """
        return softsvmpoly(alpha["lambda"], alpha["k_pol_degree"], train_x, train_y)

    @staticmethod
    def predict(x, x_train, alphas, k):
        """
        Make predictions for multiple data points using the trained soft SVM model.
        :param x: Test data (data point)
        :param x_train: Training data
        :param alphas: Learned alpha values from the soft SVM training
        :param k: Degree of the polynomial kernel
        :return: Predictions for the test data
        """
        return np.sign(
            np.inner(alphas.flatten(), [kernel(x, x_train_element, k) for x_train_element in x_train])
        )

    @staticmethod
    def evaluate_model(x_train, alphas, k, x_val, y_val):
        """
        Calculate the empirical error of the model on the validation set.

        :param x_train: Training data used to train the model
        :param alphas: Learned alpha values from the soft SVM training
        :param k: Degree of the polynomial kernel
        :param x_val: Validation feature data (vector)
        :param y_val: Validation labels (vector)
        :return: Empirical error (proportion of misclassified samples)
        """
        # Get predictions for the validation set
        predictions = []
        for vector in x_val:
            prediction = Softsvmpoly.predict(
                x=vector, x_train=x_train, alphas=alphas, k=k
            )
            predictions.append(prediction)

        predictions = np.array(predictions)
        # Calculate empirical error as the proportion of mismatches
        empirical_error = np.mean(predictions != y_val)
        return empirical_error

def kernel(x, x_tag, k):
    """
    Compute the polynomial kernel between vectors x and y with degree k
    """
    return (1 + np.dot(x, x_tag)) ** k

def create_gram_matrix(k, m, trainX):
    G = np.zeros((m, m))  # G is a symmetric matrix
    for i in range(m):
        for j in range(i, m):  # iterate over the upper triangular
            G[i, j] = kernel(trainX[i], trainX[j], k)
            if i != j:
                G[j, i] = G[i, j]
    return G


def softsvmpoly(l: float, k: int, train_x: np.array, train_y: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param k: Degree of the polynomial kernel
    :param train_x: numpy array of size (m, d) containing the training sample
    :param train_y: numpy array of size (m, 1) containing the labels of the training sample
    :return: optimal alphas
    """

    m, d = train_x.shape  # m: number of samples, d: number of features
    G = create_gram_matrix(k, m, train_x)

    epsilon = 0.001
    epsilon_matrix = np.eye(2*m) * epsilon

    # Construct matrix H as blocking matrix in size of (2*m) X (2*m)
    col1_top_h = 2 * l * G
    zero_block_m_x_m = np.zeros([m, m])
    H_blocks = np.block([[col1_top_h, zero_block_m_x_m],
                  [zero_block_m_x_m, zero_block_m_x_m]])
    H = matrix(H_blocks + epsilon_matrix)

    # Construct vector u = [0...0, 1/m...1/m] zeros are first d enters and 1/m are m next enters (size of m + m)
    u = matrix(np.append(np.zeros(m), np.ones(m) / m))

    # Construct matrix A as blocking matrix in size of 2m X 2m
    col1_top = np.zeros([m, m])
    I_m_x_m = np.eye(m)
    # For each column in the Gram matrix G, multiply every element in that column by the corresponding element in trainY.
    col1_bottom = G * train_y[:, np.newaxis]
    A_blocks = np.block([[col1_top, I_m_x_m],
                  [col1_bottom, I_m_x_m]])
    A = matrix(A_blocks + epsilon_matrix)

    # Construct vector v
    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))

    # Solve the quadratic programming problem
    sol = solvers.qp(H, u, -A, -v)

    # Extract alpha
    z = sol["x"]  # size 2m * 1  (alpha_1, ..., alpha_m, epsilon_1, ... , epsilon_m)
    alpha = np.array(z[:m])
    return alpha

def plot_training_data(x_train, y_train):
    """
    Plot the training data points in R2, color-coded by their label.

    :param x_train: numpy array of size (m, 2) containing the training sample in R2
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    """
    # Separate the data points based on their labels
    pos = x_train[y_train.flatten() == 1]
    neg = x_train[y_train.flatten() == -1]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(pos[:, 0], pos[:, 1], c="blue", marker="o", label="Positive (+1)")
    plt.scatter(neg[:, 0], neg[:, 1], c="red", marker="o", label="Negative (-1)")
    plt.title("Training Data in R2")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


def simple_test():
    # load question 2 data
    data = np.load("EX3q2_data.npz")
    trainX = data["Xtrain"]
    testX = data["Xtest"]
    trainy = data["Ytrain"]
    testy = data["Ytest"]

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(
        w, np.ndarray
    ), "The output of the function softsvmbf should be a numpy array"
    assert (
        w.shape[0] == m and w.shape[1] == 1
    ), f"The shape of the output should be ({m}, 1)"


def k_fold_cross_validation(S, A, Psi, k_folds):
    """
    Perform k-fold cross-validation on a given dataset.

    :param S: Training dataset (x_train, y_train)
    :param A: Learning algorithm with a parameter alpha
    :param Psi: A set of possible values for alpha
    :param k_folds: Number of folds
    :return: The best model trained on the entire dataset
    """
    m = len(S[0])  # number of samples in the training dataset
    indices = np.arange(m)
    folds = np.split(indices, k_folds)

    best_alpha = None
    best_error = float("inf")

    # Line 2:
    for alpha in Psi:
        errors = []

        # Line 3:
        for i in range(k_folds):
            # Line 4:
            V_indices = folds[i]  # V <- S_i
            train_indices = np.concatenate(
                [folds[j] for j in range(k_folds) if j != i]
            )  # S_tag <- S\S_i

            # Extract training and validation data
            x_train, y_train = S[0][train_indices], S[1][train_indices]
            x_val, y_val = S[0][V_indices], S[1][V_indices]

            # Line 5: Train the model
            model_opt_alphas = A.train(alpha, x_train, y_train)

            # Line 6: Evaluate the model
            error = A.evaluate_model(
                x_train=x_train,
                alphas=model_opt_alphas,
                k=k_folds,
                x_val=x_val,
                y_val=y_val,
            )
            errors.append(error)

        # Line 8:
        avg_error = np.mean(errors)
        print_str = f"Validation error: {avg_error} for {alpha}"
        final_results.append(print_str)

        # Line 10:
        if avg_error < best_error:
            best_error = avg_error
            best_alpha = alpha

    # Line 11: Train the model on the entire dataset with the best alpha
    str_best_alpha = f"Best alpha: {best_alpha}"
    final_results.append(str_best_alpha)
    opt_alphas = A.train(alpha=best_alpha, train_x=S[0], train_y=S[1])

    return SoftsvmpolyModel(opt_alphas=opt_alphas, opt_k=best_alpha["k_pol_degree"])


def predict_on_grid(model, x_min, x_max, y_min, y_max, grid_size, x_train):
    """
    Predict labels for a grid of points in the specified region.

    :param model: Trained soft SVM model
    :param x_min, x_max, y_min, y_max: Bounds of the grid in R2
    :param grid_size: The resolution of the grid
    :param x_train: numpy array of size (m, d) containing the training sample
    :return: Grid of predictions
    """
    # Create a grid of points
    x_values, y_values = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid_points = np.c_[x_values.ravel(), y_values.ravel()]

    # Predict label for each point in the grid
    predictions = np.array([model.predict(x=point, x_train=x_train) for point in grid_points])

    # Reshape predictions to match the grid
    return predictions.reshape(x_values.shape)

def q2_a(x_train, y_train):
    plot_training_data(x_train, y_train)

def q2_b(x_train, y_train, x_test, y_test):
    degrees = [2, 5, 8]
    lambdas = [1, 10, 100]
    psi = []
    for l in lambdas:
        for k_pol_degree in degrees:
            alpha = {"lambda": l, "k_pol_degree": k_pol_degree}
            psi.append(alpha)

    S = (x_train, y_train)
    model: SoftsvmpolyModel = k_fold_cross_validation(S=S, A=Softsvmpoly, Psi=psi, k_folds=5)
    print(f"optimal_alpha: {model.opt_alphas}")
    print(f"optimal_k: {model.opt_k}")
    error = model.evaluate_model(x_train=x_train, x_test=x_test, y_test=y_test)
    print(f"test error of the resulting classifier: {error}")
    print(f"Final Results:")
    for result in final_results:
        print(result)

def q2_d(x_train, y_train):
    # Define the bounds of the grid based on your data
    x_min, x_max = x_train[:, 0].min() - 0.01, x_train[:, 0].max() + 0.01
    y_min, y_max = x_train[:, 1].min() - 0.01, x_train[:, 1].max() + 0.01
    grid_size = 50

    for k in [3, 5, 8, 12]:
        alphas = Softsvmpoly.train(alpha={'lambda': 100, 'k_pol_degree': k}, train_x=x_train, train_y=y_train)
        model = SoftsvmpolyModel(opt_alphas=alphas, opt_k=k)

        # Get predictions for the grid
        grid_predictions = predict_on_grid(model, x_min, x_max, y_min, y_max, grid_size, x_train)

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_predictions, extent=(x_min, x_max, y_min, y_max), origin='lower', alpha=0.3, cmap='bwr')
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', edgecolor='k')
        plt.title(f"Decision Boundary for Polynomial Degree {k}")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

def main():
    # Load question 2 data
    data = np.load("EX3q2_data.npz")
    x_train = data["Xtrain"]
    y_train = data["Ytrain"]
    x_test = data["Xtest"]
    y_test = data["Ytest"]

    # simple_test()
    q2_a(x_train, y_train)
    q2_b(x_train, y_train, x_test, y_test)
    q2_d(x_train, y_train)


if __name__ == "__main__":

    main()