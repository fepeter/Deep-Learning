import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def eval(L):
    ret = np.zeros(L.shape)
    for i in range(len(L)):
        ret[i] = round(float(L[i]))
    return ret

def main():

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    for i in range(len(X)):
        W1 = np.array([
            [-4, -1],
            [ 9, -4],
            [ 6,  6],
            [-4,  9]
        ])
        W1 = W1.T

        B1 = np.array([
            [0,0,0,0]
        ])

        W2 = np.array([
            [   -15,     -4,     10,      0],
            [   -15,      6,    -11,     10],
            [    -5,    -12,     17,    -10]
        ])
        W2 = W2.T

        B2 = np.array([
            [0,0,0]
        ])

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