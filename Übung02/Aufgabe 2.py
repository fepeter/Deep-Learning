import numpy as np


def sigmoid(x):
    """
    :param x: 
    :return: value of sigmoid-function at x
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derived(x):
    return sigmoid(x) * (1 - sigmoid(x))


def rand_init(y, x):
    """
    positive and negative weights
    therefore: 2 * np.random.random(y, x)
    :param y: number of rows
    :param x: number of columns
    :return: y x x matrix of random weights between -1 and 1
    """
    return 2 * np.random.random([y, x]) - 1


def mse(Y, a):
    #print('Y: ', Y, '\na: ', a, '\nlen(a): ', len(a))

    error = np.zeros(Y.shape)

#    for i,j in enumerate(a):
#        error[i] = (1 / len(a)) * np.sum((Y[i] - j)**2)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[1]):
                error[i][j] += (1 / len(a)) * ((Y[i][j] - a[i][k])**2)

    #return (1 / len(a.T)) * np.sum((Y - a.T)**2)
    #print(error)
    return error


def predict():
    pass

def main():

    """
    input data
    """
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    """
    expected output data
    [OR, AND, XOR]
    """
    Y = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    epochs = 10000
    eta = 0.1
    lossOld = 1.
    eps = 0.005

    # layers[input, hidden, output]
    layers = np.array([2, 4, 3])

    W1 = rand_init(layers[0], layers[1])
    W2 = rand_init(layers[1], layers[2])

    for e in range(epochs):
        i = np.random.randint(len(X))

        #forward prop
        L1 = np.dot(X, W1) # + B1
        A1 = sigmoid(L1)
        L2 = np.dot(A1, W2)# + B2
        A2 = sigmoid(L2)

        error = Y - A2
        loss = np.mean(np.abs(error)**2)
        print('Epoch {}:\nloss: {}'.format(e, loss))
        if(np.fabs(lossOld - loss) < eps ):
            break

        delta2 = error * sigmoid_derived(L2)
        delta1 = delta2.dot(W2.T) * sigmoid_derived(L1)

        #backprop
        W2 += eta * A1.T.dot(delta2)
        W1 += eta * X.T.dot(delta1)

    #predict
    #for j in range(len(X)):
    L1 = sigmoid(np.dot(X, W1))  # + B1
    L2 = sigmoid(np.dot(L1, W2))  # + B2

    for j in range(len(X)):
        print(
            'INPUT:\t', X[j], '\n',
            'OR:\t', round(L2[j][0]), ' (', L2[j][0], ')\t(', float(Y[j][0]), ')\n',
            'AND:\t', round(L2[j][1]), ' (', L2[j][1], ')\t(', float(Y[j][1]), ')\n',
            'XOR:\t', round(L2[j][2]), ' (', L2[j][2], ')\t(', float(Y[j][2]), ')\n'
        )


if __name__ == '__main__':
    main()