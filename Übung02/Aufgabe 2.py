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

def backprop():
    pass


def predict():
    pass


def evaluate(L):
    ret = np.zeros(L.shape)
    for i in range(len(L)):
        ret[i] = round(float(L[i]))
    return ret


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

    # layers[input, hidden, output]
    layers = np.array([2, 4, 3])

    # + 1 for bias
    W1 = rand_init(layers[0], layers[1])
    #B1 = rand_init(1, layers[1])
    W2 = rand_init(layers[1], layers[2])
    #B2 = rand_init(1, layers[2])

    #b = np.atleast_2d(np.ones(X.shape[0]))
    #X = np.concatenate((b.T, X), axis=1)

    for e in range(epochs):
        i = np.random.randint(len(X))

        #forward prop
        L1 = sigmoid(np.dot(X, W1)) # + B1
        L2 = sigmoid(np.dot(L1, W2)) # + B2

        loss = mse(Y, L2)
        #loss = Y-L2
        print(loss)

        if (e % 10000) == 0:
            print(
            "Error:" + str(np.mean(np.abs(loss))))

        delta2 = loss * sigmoid_derived(L2)
        delta1 = delta2.dot(W2.T) * sigmoid_derived(L1)

        #backprop
        W2 += -eta * L1.T.dot(delta2)
        W1 += -eta * X.T.dot(delta1)

    #predict
    #for j in range(len(X)):
    L1 = sigmoid(np.dot(X, W1))  # + B1
    L2 = sigmoid(np.dot(L1, W2))  # + B2
    print(L2)
    #out = evaluate(L2.T)
    '''
    for j in len(X):
        print(
            'INPUT:\t', X[j], '\n',
            'OR:\t', out[0], ' (', L2.T[0], ')\n',
            'AND:\t', out[1], ' (', L2.T[1], ')\n',
            'XOR:\t', out[2], ' (', L2.T[2], ')\n'
        )
    '''


if __name__ == '__main__':
    main()