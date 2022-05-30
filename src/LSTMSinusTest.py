import numpy as np
import matplotlib.pyplot as plt
from LSTMNode import LSTMNode
from LSTMBackPropagator import LSTMBackPropagator
from WeightDisplay import WeightDisplay


# sinus curve from 0 - 5*2pi (= 31.4159)
# steps: 0.3 => approx. 21 steps for one 2pi cycle
# total sum of steps: 10*pi/0.5 = approx. 105
# in-sample (training data): first 83 steps
#    start: 5 (first window 0 - 4), 5th is predicted
#    moving window of 5 steps =>  80 windows
#    0 ... 4 [5]
#     1 ... 5 [6]
#      2 ... 6 [7]
#       ........
#        79 ... 83 [84]
# 1. window: 0 - 4 => predict 5th step
# 2. windoe: 1 - 5 => predict 6th step
# 3. window: 2 - 6 =? predict 7th step
# ...
#
# forecasting: step 85 till 105
#     testing windows:
#     80 ... 84 [85]
#       81 ... 85 [86]
#         ......
#         100 ... 104 [105]

def create_input_matrix(start_step, window_size, nr_windows, sinus_freq):
    matrix = np.full((window_size, nr_windows), 0.)  # initialize with float zeros
    for col in range(0, nr_windows):  # attach columns, each column is one window
        for row in range(0, window_size):
            matrix[row, col] = np.sin(sinus_freq * (start_step + col + row))
    return matrix


# value to predict
def create_values2predict(start_step, nr_values, sinus_freq):
    x = np.arange(start_step, start_step + nr_values)
    return np.sin(sinus_freq * x)


# x = np.linspace(5, 85, 80)
# y = create_values2predict(5, 80, 0.5)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()

# window size
wdw_size = 5
# number of windows for training
nr_train_wdw = 80
sinus_freq = 0.5

# trainings data
start_training = 0.
training_data = create_input_matrix(start_training, wdw_size, nr_train_wdw, sinus_freq)
start = wdw_size
training_predict_data = create_values2predict(start, nr_train_wdw, sinus_freq)
# print("training_data:")
# print(training_data)
# print("training_predict_data:")
# print(training_predict_data)
# fig, ax = plt.subplots()
# x_vals = np.arange(5, 85)
# ax.plot(x_vals, training_predict_data)
# plt.show()

# test data
start_test = 80
end_test = 100
nr_test_wdw = 21
test_data = create_input_matrix(start_test, wdw_size, nr_test_wdw, sinus_freq)
test_predict_data = create_values2predict(nr_train_wdw + wdw_size, nr_test_wdw, sinus_freq)
# print("test_data:")
# print(test_data)
# print("test_predict_data:")
# print(test_predict_data)
# fig, ax = plt.subplots()
# x_vals = np.arange(85, 106)
# ax.plot(x_vals, test_predict_data)
# plt.show()

###### trainng phase #######
x_dim = wdw_size
h_dim = 1
lstmNode = LSTMNode(x_dim, h_dim)

s_prev = 0.
h_prev = 0.
sgdParam = 0.2
num_epochs = 3
lstmBackPropagator = LSTMBackPropagator(lstmNode, sgdParam)

corr_outputs_list = []
lstmNode.store_all("../data/sinus/", "lstm_sinus_simple")

#organize plotting the input activasion
fig = plt.figure(0)
fig.suptitle('Inputactivation per Epoch')
Tot = num_epochs
Cols = 3
Rows = Tot // Cols
Rows += Tot % Cols
Position = range(1,Tot + 1)
y_counter_list = []

for epoch in range(0, num_epochs):
    lstmNode.reset_state_lists()
    y_counter_list = [] #reset index
    ax = fig.add_subplot(Rows, Cols, Position[epoch])

    for step in range(0, nr_train_wdw):
        x = training_data[:, step]
        x_scatter = training_data[:, step]

        y_counter_list.append(x_scatter)  # counter for plot index

        x = np.matrix(x).transpose()
        output = lstmNode.calculate(x, s_prev, h_prev)
        s_prev = output[0][0]
        h_prev = output[1][0]

        #convert matrix to array
        tanh_h_s_prev = np.tanh(np.asarray(s_prev).flatten()[0])

        #print(tanh_h_s_prev)

        corr_outputs = training_predict_data[step]
        corr_outputs_list.append(corr_outputs)

        plt.scatter(range(len(y_counter_list) - 1, len(y_counter_list) - 1 + 4), x_scatter[0:4],
                    color=WeightDisplay.calc_color(tanh_h_s_prev))
        plt.plot(range(len(y_counter_list) - 1, len(y_counter_list) - 1 + 4), x_scatter[0:4],
                 color="black")

        #print(h_prev[0])
        #print(type(h_prev))
    lstmBackPropagator.backpropagate(np.array(corr_outputs_list))
    lstmNode.store_all("../data/sinus/", "lstm_sinus_simple")

######## testing phase ############
output_vals = np.zeros(nr_test_wdw)
s_prev = output[0][0]
h_prev = output[1][0]
for step in range(0, nr_test_wdw):
    x = test_data[:, step]
    x = np.matrix(x).transpose()
    output = lstmNode.calculate(x, s_prev, h_prev)
    lstmNode.store_gates("../data/sinus/", "lstm_sinus_simple")
    s_prev = output[0][0]
    h_prev = output[1][0]
    # print(output)
    output_vals[step] = output[1][0]

# print("test data:")
# print(test_data)
# print("test predict data:")
# print(test_predict_data)
# print(output_vals)

fig2, ax = plt.subplots()
x_vals = np.arange(85, 85 + nr_test_wdw)
ax.plot(x_vals, output_vals, 'r')
ax.plot(x_vals, test_predict_data, 'g')
plt.show()

