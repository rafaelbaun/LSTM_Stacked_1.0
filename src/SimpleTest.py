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
h_dim = 1

x = np.array([[  1, 0.5],
              [  2, 3]])
s_prev = 0
h_prev = 0

lstmNode = LSTMNode(x_dim, h_dim)
lstmNode.wg = np.array([0.45, 0.25])
lstmNode.ug = np.array([0.15])
lstmNode.bg = 0.2
lstmNode.wi = np.array([0.95, 0.8])
lstmNode.ui = np.array([0.8])
lstmNode.bi = 0.65
lstmNode.wf = np.array([0.7, 0.45])
lstmNode.uf = np.array([0.1])
lstmNode.bf = 0.15
lstmNode.wo = np.array([0.6, 0.4])
lstmNode.uo = np.array([0.25])
lstmNode.bo = 0.1

for n in range(num_steps):
    output = lstmNode.calculate(x[:, n], s_prev, h_prev)
    print(output)
    s_prev = output[0]
    h_prev = output[1]
    print('prev state: ' + str(s_prev))
    print('prev h: ' + str(h_prev))
    if n == 0:
        checkit('state_0', s_prev, 0.78572)
        checkit('h_0', h_prev, 0.53631)
    else:
        checkit('state_1', s_prev, 1.5176)
        checkit('h_1', h_prev, 0.77197)

print('--------- start backpropagation -----------')
sgdParam = 0.1
lstmBackPropagator = LSTMBackPropagator(lstmNode, sgdParam)
corr_outputs = np.array([0.5, 1.25])
lstmBackPropagator.backpropagate(corr_outputs)
print('******** weights w ********')
# 0.452670.25922
print(lstmNode.wg)
# 0.950220.80067
print(lstmNode.wi)
# 0.700310.45189
print(lstmNode.wf)
# 0.602590.41626
print(lstmNode.wo)
print('******** weights u ********')
print(lstmNode.ug)
print(lstmNode.ui)
print(lstmNode.uf)
print(lstmNode.uo)
checkit('ug', lstmNode.ug, 0.15104)
checkit('ui', lstmNode.ui, 0.80006)
checkit('uf', lstmNode.uf, 0.10034)
checkit('uo', lstmNode.uo, 0.25297)
print('******** biases b ********')
print(lstmNode.bg)
print(lstmNode.bi)
print(lstmNode.bf)
print(lstmNode.bo)
checkit('bg', lstmNode.bg, 0.20364)
checkit('bi', lstmNode.bi, 0.65028)
checkit('bf', lstmNode.bf, 0.15063)
checkit('bo', lstmNode.bo, 0.10536)

