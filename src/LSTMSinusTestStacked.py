import numpy as np
import matplotlib.pyplot as plt
from LSTMNode import LSTMNode
from LSTMStackedBackPropagator import LSTMStackedBackPropagator
from  WeightDisplay import WeightDisplay

import matplotlib.pyplot as plt



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

# test data
start_test = 80
end_test = 100
nr_test_wdw = 21
test_data = create_input_matrix(start_test, wdw_size, nr_test_wdw, sinus_freq)
test_predict_data = create_values2predict(nr_train_wdw + wdw_size, nr_test_wdw, sinus_freq)

###### trainng phase #######
x_dim = wdw_size
# good results:
# (sgdParam, epochs, num lstm):
# (0.02, 18, 2)
#
sgdParam = 0.02
num_epochs = 18
num_stacked_lstm = 2

lstmNode_list = []
# all lstm cells except the last one
h_dim = 5
for n in range(num_stacked_lstm - 1):
    lstmNode_list.append(LSTMNode(x_dim, h_dim))
# last lstm cell
h_dim = 1
lstmNode_list.append(LSTMNode(x_dim, h_dim))

for k in range(num_stacked_lstm):
    lstmNode_list[k].store_all("../data/stacked/", "lstm_cell_" + str(k))

s_prev_list = [] # list of s_prev per lstm cell except last cell
h_prev_list = [] # list of h_prev per lstm cell except last cell

for j in range(num_stacked_lstm - 1): # all but last cell
    s_prev_list.append(np.zeros(x_dim))
    h_prev_list.append(np.zeros(x_dim))

# s_prev, h_prev of last cell are 1-dimensional
s_prev_lastcell = 0.
h_prev_lastcell = 0.

output = None
corr_outputs_list = []
lstmStackedBackPropagator = LSTMStackedBackPropagator(lstmNode_list, sgdParam)

#organize plotting the input activasion
fig = plt.figure(0)
fig.suptitle('Input Activation per Epoch')
Tot = num_epochs
Cols = 3
Rows = Tot // Cols
Rows += Tot % Cols
Position = range(1,Tot + 1)
y_counter_list = []


for epoch in range(0, num_epochs):
    y_counter_list = [] # reset index

    # add subplot to figure
    ax = fig.add_subplot(Rows, Cols, Position[epoch])


    #ax.set_title("Epoch " + str(epoch))
    # reset state lists of all lstm cells
    for j in range(num_stacked_lstm):
        lstmNode_list[j].reset_state_lists()

    for step in range(0, nr_train_wdw): # rolling window
        x = training_data[:, step]
        y_counter_list.append(x) #counter for plot index
        # calculate thru all lstm nodes
        for lstm_idx in range(num_stacked_lstm):
            if lstm_idx == 0: # take input from sinus wave
                if lstm_idx == (num_stacked_lstm - 1): # the only and last cell
                    output = lstmNode_list[lstm_idx].calculate(x, s_prev_lastcell, h_prev_lastcell)
                    s_prev_lastcell = output[0]
                    h_prev_lastcell = output[1]
                else:
                    output = lstmNode_list[lstm_idx].calculate(x, s_prev_list[lstm_idx], h_prev_list[lstm_idx])

            else: # take output from previous lstm cell as input
                if lstm_idx == (num_stacked_lstm - 1): #last cell
                    output = lstmNode_list[lstm_idx].calculate(output[1], s_prev_lastcell, h_prev_lastcell)
                    s_prev_lastcell = output[0]
                    h_prev_lastcell = output[1]

                    plt.scatter(range(len(y_counter_list) - 1,len(y_counter_list) - 1 + 4), x[0:4],
                                color=WeightDisplay.calc_color(np.tanh(output[0])))
                    plt.plot(range(len(y_counter_list) - 1,len(y_counter_list) - 1 + 4), x[0:4],
                             color="black")

                else:
                    output = lstmNode_list[lstm_idx].calculate(output[1], s_prev_list[lstm_idx], h_prev_list[lstm_idx])
                    s_prev_list[lstm_idx] = output[0]
                    h_prev_list[lstm_idx] = output[1]

        corr_outputs = training_predict_data[step]
        corr_outputs_list.append(corr_outputs)

    lstmStackedBackPropagator.backpropagate(np.array(corr_outputs_list))
    for cell_idx in range(num_stacked_lstm):
        lstmNode_list[cell_idx].store_all("../data/stacked/", "lstm_cell_" + str(cell_idx))


#plt.show()
######## testing phase ############
output_vals = np.zeros(nr_test_wdw)
for j in range(num_stacked_lstm):
    lstmNode_list[j].reset_state_lists()

for step in range(0, nr_test_wdw):
    x = test_data[:, step]

    # calculate thru all lstm nodes
    for lstm_idx in range(num_stacked_lstm):
        if lstm_idx == 0:  # take input from sinus wave
            if lstm_idx == (num_stacked_lstm - 1):  # the only and last cell
                output = lstmNode_list[lstm_idx].calculate(x, s_prev_lastcell, h_prev_lastcell)
                s_prev_lastcell = output[0]
                h_prev_lastcell = output[1]

            else:
                output = lstmNode_list[lstm_idx].calculate(x, s_prev_list[lstm_idx], h_prev_list[lstm_idx])

        else:  # take output from previous lstm cell as input
            if lstm_idx == (num_stacked_lstm - 1):  # last cell
                output = lstmNode_list[lstm_idx].calculate(output[1], s_prev_lastcell, h_prev_lastcell)
                s_prev_lastcell = output[0]
                h_prev_lastcell = output[1]


            else:
                output = lstmNode_list[lstm_idx].calculate(output[1], s_prev_list[lstm_idx], h_prev_list[lstm_idx])
                s_prev_list[lstm_idx] = output[0]
                h_prev_list[lstm_idx] = output[1]



    output_vals[step] = output[1]

fig2, ax = plt.subplots()
x_vals = np.arange(85, 85 + nr_test_wdw)
ax.plot(x_vals, output_vals, 'r', label = "Test")
ax.plot(x_vals, test_predict_data, 'g', label= "Sinus")
#plt.title("Test vs Sinuskurve")
plt.legend(loc='best')
plt.show()
