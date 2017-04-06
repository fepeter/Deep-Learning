import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(A, B):
    return ((A - B) ** 2).mean(axis=None)


def getRandomArray(hight, width, rangeLow, rangeHigh):
    a = np.random.uniform(rangeLow, rangeHigh, size=hight * width)
    return a.reshape(width, hight)


def eval(L):
    ret = np.zeros(L.shape)
    for i in range(len(L)):
        ret[i] = round(float(L[i]))
    return ret


def main():
    # Eingabe sim
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    # Trainingsdata
    Y = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])

    # Create random weights and biases
    W1 = getRandomArray(4, 2, -0.1, 0.1)
    B1 = getRandomArray(4, 1, -0.1, 0.1)

    W2 = getRandomArray(3, 4, -0.1, 0.1)
    B2 = getRandomArray(3, 1, -0.1, 0.1)

    print("======================================================"
          "===  Forward "
          "======================================================")
    for i in range(len(X)):
        L1 = sigmoid(np.dot(X[i], W1) + B1)
        L2 = sigmoid(np.dot(L1, W2) + B2)

        out = eval(L2.T)

        print(
            'INPUT:\t', X[i], '\n',
            'OR:\t', out[0], ' (', L2.T[0], ')\n',
            'AND:\t', out[1], ' (', L2.T[1], ')\n',
            'XOR:\t', out[2], ' (', L2.T[2], ')\n'
        )

    # backprob
    print("======================================================"
          "===  Backprob"
          "======================================================")
    for i in reversed(range(len(X))):
        # ... TODO backprob implementieren
        print(i)

    print("======================================================"
          "===  Forward again"
          "======================================================")
    for i in range(len(X)):
        # neuer Vorwärtslauf mit korregierten Gewichten und Bias
        L1 = sigmoid(np.dot(X[i], W1) + B1)
        L2 = sigmoid(np.dot(L1, W2) + B2)

        out = eval(L2.T)

        print(
            'INPUT:\t', X[i], '\n',
            'OR:\t', out[0], ' (', L2.T[0], ')\n',
            'AND:\t', out[1], ' (', L2.T[1], ')\n',
            'XOR:\t', out[2], ' (', L2.T[2], ')\n'
        )


if __name__ == '__main__':
    main()
