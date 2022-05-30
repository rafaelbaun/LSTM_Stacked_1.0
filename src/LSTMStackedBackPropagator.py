import numpy as np

class LSTMStackedBackPropagator:
    delta_wg_list = []
    delta_wi_list = []
    delta_wo_list = []
    delta_wf_list = []
    delta_ug_list = []
    delta_ui_list = []
    delta_uo_list = []
    delta_uf_list = []
    delta_bg_list = []
    delta_bi_list = []
    delta_bo_list = []
    delta_bf_list = []

    def __init__(self, lstmNode_list, sgd_param):
        self.lstmNode_list = lstmNode_list
        self.sgd_param = sgd_param
        self.num_cell = len(lstmNode_list)
        for cell_idx in range(self.num_cell):
            self.delta_wg_list.append(np.zeros(self.lstmNode_list[cell_idx].x_dim))
            self.delta_wi_list.append(np.zeros(self.lstmNode_list[cell_idx].x_dim))
            self.delta_wo_list.append(np.zeros(self.lstmNode_list[cell_idx].x_dim))
            self.delta_wf_list.append(np.zeros(self.lstmNode_list[cell_idx].x_dim))
            self.delta_ug_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_ui_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_uo_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_uf_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_bg_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_bi_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_bo_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))
            self.delta_bf_list.append(np.zeros(self.lstmNode_list[cell_idx].h_dim))

    def sigmoid_derivative(self, values) -> float:
        return values * (1 - values)

    def tanh_derivative(self, values) -> float:
        return 1. - np.tanh(values) ** 2

    def initialize_diff_state_lists(self, cell_idx, num_steps):
        dim_corr_outp = self.lstmNode_list[cell_idx].h_dim
        if dim_corr_outp == 1:  # the last cell has 1-dimensional output
            self.diff_state_h_list.append(np.zeros(num_steps))
            self.diff_state_s_list.append(np.zeros(num_steps))
            self.diff_state_i_list.append(np.zeros(num_steps))
            self.diff_state_g_list.append(np.zeros(num_steps))
            self.diff_state_f_list.append(np.zeros(num_steps))
            self.diff_state_o_list.append(np.zeros(num_steps))
            self.delta_h_list.append(np.zeros(num_steps))
            self.diff_x_list.append(np.zeros((self.lstmNode_list[cell_idx].x_dim, num_steps)))
            self.last_delta_h_list.append(np.zeros(dim_corr_outp))
            self.state_f_next_list.append(None)
        else:
            self.diff_state_h_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_state_s_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_state_i_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_state_g_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_state_f_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_state_o_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.delta_h_list.append(np.zeros((dim_corr_outp, num_steps)))
            self.diff_x_list.append(np.zeros((self.lstmNode_list[cell_idx].x_dim, num_steps)))
            self.last_delta_h_list.append(np.zeros(dim_corr_outp))
            self.state_f_next_list.append(None)

    def backpropagate(self, corr_outputs):
        num_steps = self.lstmNode_list[0].num_steps
        self.diff_state_h_list = []
        self.diff_state_s_list = []
        self.diff_state_i_list = []
        self.diff_state_g_list = []
        self.diff_state_f_list = []
        self.diff_state_o_list = []
        self.delta_h_list = []
        self.diff_x_list = []
        self.last_delta_h_list = []
        self.state_f_next_list = []
        
        # initialize diff cell lists (last cell has 1-dim. output!)
        for cell_idx in range(self.num_cell):
            self.initialize_diff_state_lists(cell_idx, num_steps)

        for step in reversed(range(num_steps)):
            corr_outp_step = corr_outputs[step]
            # x_input = self.lstmNode_list[0].x_list[step]
            for cell_idx in reversed(range(self.num_cell)):
                dim_corr_outp = self.lstmNode_list[cell_idx].h_dim
                state_s = self.lstmNode_list[cell_idx].state_s_list[step] # has dimension dim_corr_outp
                state_h = self.lstmNode_list[cell_idx].state_h_list[step]
                state_o = self.lstmNode_list[cell_idx].state_o_list[step]
                state_g = self.lstmNode_list[cell_idx].state_g_list[step]
                state_f = self.lstmNode_list[cell_idx].state_f_list[step]
                state_i = self.lstmNode_list[cell_idx].state_i_list[step]

                if dim_corr_outp == 1: # the last cell
                    if step == (num_steps - 1):
                        self.diff_state_h_list[cell_idx][step] = state_h - corr_outp_step
                        self.diff_state_s_list[cell_idx][step] = self.diff_state_h_list[cell_idx][step] * state_o * self.tanh_derivative(state_s)
                    else:
                        state_f_next = self.lstmNode_list[cell_idx].state_f_list[step + 1]
                        self.diff_state_h_list[cell_idx][step] = state_h - corr_outp_step + self.last_delta_h_list[cell_idx]
                        self.diff_state_s_list[cell_idx][step] = self.diff_state_h_list[cell_idx][step] * state_o * self.tanh_derivative(state_s) + \
                                                     self.diff_state_s_list[cell_idx][step + 1] * state_f_next

                    self.diff_state_g_list[cell_idx][step] = self.diff_state_s_list[cell_idx][step] * state_i * (1 - state_g ** 2)
                    self.diff_state_i_list[cell_idx][step] = self.diff_state_s_list[cell_idx][step] * state_g * self.sigmoid_derivative(state_i)

                    if step == 0:
                        self.diff_state_f_list[cell_idx][step] = np.zeros(dim_corr_outp)
                    else:
                        state_s_prev = self.lstmNode_list[cell_idx].state_s_list[step - 1]
                        self.diff_state_f_list[cell_idx][step] = self.diff_state_s_list[cell_idx][step] * state_s_prev * self.sigmoid_derivative(state_f)

                    self.diff_state_o_list[cell_idx][step] = self.diff_state_h_list[cell_idx][step] * np.tanh(state_s) * self.sigmoid_derivative(state_o)
                    self.diff_x_list[cell_idx][:, step] = np.dot(self.diff_state_g_list[cell_idx][step], self.lstmNode_list[cell_idx].wg) + np.dot(self.diff_state_i_list[cell_idx][step], self.lstmNode_list[cell_idx].wi) + \
                                           np.dot(self.diff_state_f_list[cell_idx][step], self.lstmNode_list[cell_idx].wf) + np.dot(self.diff_state_o_list[cell_idx][step], self.lstmNode_list[cell_idx].wo)
                    if step > 0:
                        self.delta_h_list[cell_idx][step - 1] = np.dot(self.diff_state_g_list[cell_idx][step], self.lstmNode_list[cell_idx].ug) + np.dot(self.diff_state_i_list[cell_idx][step], self.lstmNode_list[cell_idx].ui) + \
                                                    np.dot(self.diff_state_f_list[cell_idx][step], self.lstmNode_list[cell_idx].uf) + np.dot(self.diff_state_o_list[cell_idx][step], self.lstmNode_list[cell_idx].uo)
                        self.last_delta_h_list[cell_idx] = np.dot(self.diff_state_g_list[cell_idx][step], self.lstmNode_list[cell_idx].ug) + np.dot(self.diff_state_i_list[cell_idx][step], self.lstmNode_list[cell_idx].ui) + \
                                       np.dot(self.diff_state_f_list[cell_idx][step], self.lstmNode_list[cell_idx].uf) + np.dot(self.diff_state_o_list[cell_idx][step], self.lstmNode_list[cell_idx].uo)
                else:
                    if step == (num_steps - 1):
                        self.diff_state_h_list[cell_idx][:, step] = state_h + self.diff_x_list[cell_idx + 1][:, step]
                        self.diff_state_s_list[cell_idx][:, step] = self.diff_state_h_list[cell_idx][:, step] * state_o * self.tanh_derivative(state_s)
                    else:
                        state_f_next = self.lstmNode_list[cell_idx].state_f_list[step + 1]
                        self.diff_state_h_list[cell_idx][:, step] = state_h  - self.last_delta_h_list[cell_idx] + self.diff_x_list[cell_idx + 1][:, step]
                        self.diff_state_s_list[cell_idx][:, step] = self.diff_state_h_list[cell_idx][:, step] * state_o * self.tanh_derivative(state_s) + \
                                                  self.diff_state_s_list[cell_idx][:, step + 1] * state_f_next

                    self.diff_state_g_list[cell_idx][:, step] = self.diff_state_s_list[cell_idx][:, step] * state_i * ( 1 - state_g**2)
                    self.diff_state_i_list[cell_idx][:, step] = self.diff_state_s_list[cell_idx][:, step] * state_g * self.sigmoid_derivative(state_i)

                    if step == 0:
                        self.diff_state_f_list[cell_idx][:, step] = np.zeros(dim_corr_outp)
                    else:
                        state_s_prev = self.lstmNode_list[cell_idx].state_s_list[step-1]
                        self.diff_state_f_list[cell_idx][:, step] = self.diff_state_s_list[cell_idx][:, step] * state_s_prev * self.sigmoid_derivative(state_f)

                    self.diff_state_o_list[cell_idx][:, step] = self.diff_state_h_list[cell_idx][:, step] * np.tanh(state_s) * self.sigmoid_derivative(state_o)
                    self.diff_x_list[cell_idx][:, step] = np.dot(self.diff_state_g_list[cell_idx][:, step], self.lstmNode_list[cell_idx].wg) + np.dot(self.diff_state_i_list[cell_idx][:, step], self.lstmNode_list[cell_idx].wi) + \
                                        np.dot(self.diff_state_f_list[cell_idx][:, step], self.lstmNode_list[cell_idx].wf) + np.dot(self.diff_state_o_list[cell_idx][:, step], self.lstmNode_list[cell_idx].wo)
                    if step > 0:
                        self.delta_h_list[cell_idx][:, step - 1] = np.dot(self.diff_state_g_list[cell_idx][:, step], self.lstmNode_list[cell_idx].ug) + np.dot(self.diff_state_i_list[cell_idx][:, step], self.lstmNode_list[cell_idx].ui) + \
                                                 np.dot(self.diff_state_f_list[cell_idx][:, step], self.lstmNode_list[cell_idx].uf) + np.dot(self.diff_state_o_list[cell_idx][:, step], self.lstmNode_list[cell_idx].uo)
                        last_delta_h = np.dot(self.diff_state_g_list[cell_idx][:, step], self.lstmNode_list[cell_idx].ug) + np.dot(self.diff_state_i_list[cell_idx][:, step], self.lstmNode_list[cell_idx].ui) + \
                                       np.dot(self.diff_state_f_list[cell_idx][:, step], self.lstmNode_list[cell_idx].uf) + np.dot(self.diff_state_o_list[cell_idx][:, step], self.lstmNode_list[cell_idx].uo)
        self.calculate_weights(num_steps)
        return self.lstmNode_list

    def calculate_weights(self, num_steps):
        # calculate weight corrections
        for cell_idx in reversed(range(self.num_cell)):
            dim_corr_outp = self.lstmNode_list[cell_idx].h_dim
            if dim_corr_outp == 1:
                for n in reversed(range(num_steps)):
                    self.delta_wg_list[cell_idx] = self.delta_wg_list[cell_idx] + (self.diff_state_g_list[cell_idx][n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wi_list[cell_idx] = self.delta_wi_list[cell_idx] + (self.diff_state_i_list[cell_idx][n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wo_list[cell_idx] = self.delta_wo_list[cell_idx] + (self.diff_state_o_list[cell_idx][n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wf_list[cell_idx] = self.delta_wf_list[cell_idx] + (self.diff_state_f_list[cell_idx][n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                for n in reversed(range(num_steps - 1)):
                    self.delta_ug_list[cell_idx] = self.delta_ug_list[cell_idx] + (self.diff_state_g_list[cell_idx][n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_ui_list[cell_idx] = self.delta_ui_list[cell_idx] + (self.diff_state_i_list[cell_idx][n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_uo_list[cell_idx] = self.delta_uo_list[cell_idx] + (self.diff_state_o_list[cell_idx][n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_uf_list[cell_idx] = self.delta_uf_list[cell_idx] + (self.diff_state_f_list[cell_idx][n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                for n in reversed(range(num_steps)):
                    self.delta_bg_list[cell_idx] = self.delta_bg_list[cell_idx] + self.diff_state_g_list[cell_idx][n]
                    self.delta_bi_list[cell_idx] = self.delta_bi_list[cell_idx] + self.diff_state_i_list[cell_idx][n]
                    self.delta_bo_list[cell_idx] = self.delta_bo_list[cell_idx] + self.diff_state_o_list[cell_idx][n]
                    self.delta_bf_list[cell_idx] = self.delta_bf_list[cell_idx] + self.diff_state_f_list[cell_idx][n]
            else:
                for n in reversed(range(num_steps)):
                    self.delta_wg_list[cell_idx] = self.delta_wg_list[cell_idx] + (self.diff_state_g_list[cell_idx][:, n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wi_list[cell_idx] = self.delta_wi_list[cell_idx] + (self.diff_state_i_list[cell_idx][:, n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wo_list[cell_idx] = self.delta_wo_list[cell_idx] + (self.diff_state_o_list[cell_idx][:, n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                    self.delta_wf_list[cell_idx] = self.delta_wf_list[cell_idx] + (self.diff_state_f_list[cell_idx][:, n] * self.lstmNode_list[cell_idx].x_list[n]).transpose()
                for n in reversed(range(num_steps - 1)):
                    self.delta_ug_list[cell_idx] = self.delta_ug_list[cell_idx] + (self.diff_state_g_list[cell_idx][:, n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_ui_list[cell_idx] = self.delta_ui_list[cell_idx] + (self.diff_state_i_list[cell_idx][:, n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_uo_list[cell_idx] = self.delta_uo_list[cell_idx] + (self.diff_state_o_list[cell_idx][:, n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                    self.delta_uf_list[cell_idx] = self.delta_uf_list[cell_idx] + (self.diff_state_f_list[cell_idx][:, n + 1] * self.lstmNode_list[cell_idx].state_h_list[n]).transpose()
                for n in reversed(range(num_steps)):
                    self.delta_bg_list[cell_idx] = self.delta_bg_list[cell_idx] + self.diff_state_g_list[cell_idx][:, n]
                    self.delta_bi_list[cell_idx] = self.delta_bi_list[cell_idx] + self.diff_state_i_list[cell_idx][:, n]
                    self.delta_bo_list[cell_idx] = self.delta_bo_list[cell_idx] + self.diff_state_o_list[cell_idx][:, n]
                    self.delta_bf_list[cell_idx] = self.delta_bf_list[cell_idx] + self.diff_state_f_list[cell_idx][:, n]

            self.lstmNode_list[cell_idx].wg = self.lstmNode_list[cell_idx].wg - self.sgd_param * self.delta_wg_list[cell_idx]
            self.lstmNode_list[cell_idx].wi = self.lstmNode_list[cell_idx].wi - self.sgd_param * self.delta_wi_list[cell_idx]
            self.lstmNode_list[cell_idx].wo = self.lstmNode_list[cell_idx].wo - self.sgd_param * self.delta_wo_list[cell_idx]
            self.lstmNode_list[cell_idx].wf = self.lstmNode_list[cell_idx].wf - self.sgd_param * self.delta_wf_list[cell_idx]
    
            self.lstmNode_list[cell_idx].ug = self.lstmNode_list[cell_idx].ug - self.sgd_param * self.delta_ug_list[cell_idx]
            self.lstmNode_list[cell_idx].ui = self.lstmNode_list[cell_idx].ui - self.sgd_param * self.delta_ui_list[cell_idx]
            self.lstmNode_list[cell_idx].uo = self.lstmNode_list[cell_idx].uo - self.sgd_param * self.delta_uo_list[cell_idx]
            self.lstmNode_list[cell_idx].uf = self.lstmNode_list[cell_idx].uf - self.sgd_param * self.delta_uf_list[cell_idx]
    
            self.lstmNode_list[cell_idx].bg = self.lstmNode_list[cell_idx].bg - self.sgd_param * self.delta_bg_list[cell_idx]
            self.lstmNode_list[cell_idx].bi = self.lstmNode_list[cell_idx].bi - self.sgd_param * self.delta_bi_list[cell_idx]
            self.lstmNode_list[cell_idx].bo = self.lstmNode_list[cell_idx].bo - self.sgd_param * self.delta_bo_list[cell_idx]
            self.lstmNode_list[cell_idx].bf = self.lstmNode_list[cell_idx].bf - self.sgd_param * self.delta_bf_list[cell_idx]
