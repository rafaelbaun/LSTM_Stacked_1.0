import numpy as np
from LSTMNode import LSTMNode
from LSTMBackPropagator import LSTMBackPropagator


def checkit(text, val, corr_val):
    if abs(val - corr_val) < 0.001:
        print(text + " is OK.")
    else:
        print(text + ' is NOT OK.')


num_steps = 2
x_dim = 2
h_dim = 2

x = np.array([[  1, 0.5],
              [  2, 3]])
s_prev = np.array([0., 0.])
h_prev = np.array([0., 0.])

lstmNode = LSTMNode(x_dim, h_dim)
lstmNode.wg = np.array([[0.3, 0.22],
                       [0.15, 0.49]])
lstmNode.ug = np.array([[0.19, 0.51],
                       [0.26, 0.40]])
lstmNode.bg = np.array([0.2, 0.3])

lstmNode.wi = np.array([[0.09, 0.31],
                       [0.2, 0.18]])
lstmNode.ui = np.array([[0.01, 0.7],
                       [0.9, 0.6]])
lstmNode.bi = np.array([0.12, 0.23])

lstmNode.wf = np.array([[0.3, 0.1],
                       [0.2, 0.3]])
lstmNode.uf = np.array([[0.15, 0.32],
                       [0.27, 0.18]])
lstmNode.bf = np.array([0.12, 0.19])

lstmNode.wo = np.array([[0.18, 0.14],
                       [0.27, 0.29]])
lstmNode.uo = np.array([[0.3, 0.25],
                       [0.17, 0.2]])
lstmNode.bo = np.array([0.1, 0.13])

for n in range(num_steps):
    output = lstmNode.calculate(x[:, n], s_prev, h_prev)
    print(output)
    s_prev = output[0]
    h_prev = output[1]
    print('prev state: ')
    print(s_prev)
    print('prev h: ')
    print(h_prev)

print('--------- start backpropagation -----------')
sgdParam = 0.1
lstmBackPropagator = LSTMBackPropagator(lstmNode, sgdParam)
corr_outputs = np.array([[0.9, 0.6],
                        [0.7, 0.5]])
lstmBackPropagator.backpropagate(corr_outputs)

