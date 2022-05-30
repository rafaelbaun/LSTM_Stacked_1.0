import numpy as np

class LSTMBackPropagator:
    def __init__(self, lstmNode, sgd_param):
        self.lstmNode = lstmNode
        self.sgd_param = sgd_param
        self.delta_wg = np.zeros(self.lstmNode.x_dim)
        self.delta_wi = np.zeros(self.lstmNode.x_dim)
        self.delta_wo = np.zeros(self.lstmNode.x_dim)
        self.delta_wf = np.zeros(self.lstmNode.x_dim)
        self.delta_ug = np.zeros(self.lstmNode.h_dim)
        self.delta_ui = np.zeros(self.lstmNode.h_dim)
        self.delta_uo = np.zeros(self.lstmNode.h_dim)
        self.delta_uf = np.zeros(self.lstmNode.h_dim)
        self.delta_bg = 0.
        self.delta_bi = 0.
        self.delta_bo = 0.
        self.delta_bf = 0.

    def sigmoid_derivative(self, values) -> float:
        return values * (1 - values)

    def tanh_derivative(self, values) -> float:
        return 1. - np.tanh(values) ** 2

    def backpropagate(self, corr_outputs):
        num_steps = self.lstmNode.num_steps
        arr_dim = corr_outputs.ndim
        if arr_dim == 1:
            dim_corr_outp = 1
            self.diff_state_h = np.zeros(num_steps)
            self.diff_state_s = np.zeros(num_steps)
            self.diff_state_i = np.zeros(num_steps)
            self.diff_state_g = np.zeros(num_steps)
            self.diff_state_f = np.zeros(num_steps)
            self.diff_state_o = np.zeros(num_steps)
            self.delta_h = np.zeros(num_steps)
        else:
            dim_corr_outp = len(corr_outputs[:, 0])
            self.diff_state_h = np.zeros((dim_corr_outp, num_steps))
            self.diff_state_s = np.zeros((dim_corr_outp, num_steps))
            self.diff_state_i = np.zeros((dim_corr_outp, num_steps))
            self.diff_state_g = np.zeros((dim_corr_outp, num_steps))
            self.diff_state_f = np.zeros((dim_corr_outp, num_steps))
            self.diff_state_o = np.zeros((dim_corr_outp, num_steps))
            self.delta_h      = np.zeros((dim_corr_outp, num_steps))
        self.diff_x       = np.zeros((self.lstmNode.x_dim, num_steps))
        last_delta_h = np.zeros(dim_corr_outp)
        self.state_f_next = None

        for step in reversed(range(num_steps)):
            if dim_corr_outp == 1:
                corr_outp_step = corr_outputs[step]
            else:
                corr_outp_step = corr_outputs[:, step]
            state_s = self.lstmNode.state_s_list[step] # has dimension dim_corr_outp
            state_h = self.lstmNode.state_h_list[step]
            state_o = self.lstmNode.state_o_list[step]
            state_g = self.lstmNode.state_g_list[step]
            state_f = self.lstmNode.state_f_list[step]
            state_i = self.lstmNode.state_i_list[step]

            if dim_corr_outp == 1:
                if step == (num_steps - 1):
                    self.diff_state_h[step] = state_h - corr_outp_step
                    self.diff_state_s[step] = self.diff_state_h[step] * state_o * self.tanh_derivative(state_s)
                else:
                    state_f_next = self.lstmNode.state_f_list[step + 1]
                    self.diff_state_h[step] = state_h - corr_outp_step + last_delta_h
                    self.diff_state_s[step] = self.diff_state_h[step] * state_o * self.tanh_derivative(state_s) + \
                                                 self.diff_state_s[step + 1] * state_f_next
                self.diff_state_g[step] = self.diff_state_s[step] * state_i * (1 - state_g ** 2)
                self.diff_state_i[step] = self.diff_state_s[step] * state_g * self.sigmoid_derivative(state_i)
                if step == 0:
                    self.diff_state_f[step] = np.zeros(dim_corr_outp)
                else:
                    state_s_prev = self.lstmNode.state_s_list[step - 1]
                    self.diff_state_f[step] = self.diff_state_s[step] * state_s_prev * self.sigmoid_derivative(state_f)
                self.diff_state_o[step] = self.diff_state_h[step] * np.tanh(state_s) * self.sigmoid_derivative(state_o)
                self.diff_x[:, step] = np.dot(self.diff_state_g[step], self.lstmNode.wg) + np.dot(self.diff_state_i[step], self.lstmNode.wi) + \
                                       np.dot(self.diff_state_f[step], self.lstmNode.wf) + np.dot(self.diff_state_o[step], self.lstmNode.wo)
                if step > 0:
                    self.delta_h[step - 1] = np.dot(self.diff_state_g[step], self.lstmNode.ug) + np.dot(self.diff_state_i[step], self.lstmNode.ui) + \
                                                np.dot(self.diff_state_f[step], self.lstmNode.uf) + np.dot(self.diff_state_o[step], self.lstmNode.uo)
                    last_delta_h = np.dot(self.diff_state_g[step], self.lstmNode.ug) + np.dot(self.diff_state_i[step], self.lstmNode.ui) + \
                                   np.dot(self.diff_state_f[step], self.lstmNode.uf) + np.dot(self.diff_state_o[step], self.lstmNode.uo)
            else:
                if step == (num_steps - 1):
                    self.diff_state_h[:, step] = state_h - corr_outp_step
                    self.diff_state_s[:, step] = self.diff_state_h[:, step] * state_o * self.tanh_derivative(state_s)
                else:
                    state_f_next = self.lstmNode.state_f_list[step + 1]
                    self.diff_state_h[:, step] = state_h - corr_outp_step + last_delta_h
                    self.diff_state_s[:, step] = self.diff_state_h[:, step] * state_o * self.tanh_derivative(state_s) + \
                                              self.diff_state_s[:, step + 1] * state_f_next
                self.diff_state_g[:, step] = self.diff_state_s[:, step] * state_i * ( 1 - state_g**2)
                self.diff_state_i[:, step] = self.diff_state_s[:, step] * state_g * self.sigmoid_derivative(state_i)
                print("step = " + str(step))
                print(self.diff_state_h[:, step])
                print(self.diff_state_s[:, step])
                print(self.diff_state_g[:, step])
                print(self.diff_state_i[:, step])
                if step == 0:
                    self.diff_state_f[:, step] = np.zeros(dim_corr_outp)
                else:
                    state_s_prev = self.lstmNode.state_s_list[step-1]
                    self.diff_state_f[:, step] = self.diff_state_s[:, step] * state_s_prev * self.sigmoid_derivative(state_f)
                self.diff_state_o[:, step] = self.diff_state_h[:, step] * np.tanh(state_s) * self.sigmoid_derivative(state_o)
                print(self.diff_state_f[:, step])
                print(self.diff_state_o[:, step])
                self.diff_x[:, step] = np.dot(self.diff_state_g[:, step], self.lstmNode.wg) + np.dot(self.diff_state_i[:, step], self.lstmNode.wi) + \
                                    np.dot(self.diff_state_f[:, step], self.lstmNode.wf) + np.dot(self.diff_state_o[:, step], self.lstmNode.wo)
                print(self.diff_x[:, step])
                if step > 0:
                    self.delta_h[:, step - 1] = np.dot(self.diff_state_g[:, step], self.lstmNode.ug) + np.dot(self.diff_state_i[:, step], self.lstmNode.ui) + \
                                             np.dot(self.diff_state_f[:, step], self.lstmNode.uf) + np.dot(self.diff_state_o[:, step], self.lstmNode.uo)
                    last_delta_h = np.dot(self.diff_state_g[:, step], self.lstmNode.ug) + np.dot(self.diff_state_i[:, step], self.lstmNode.ui) + \
                                   np.dot(self.diff_state_f[:, step], self.lstmNode.uf) + np.dot(self.diff_state_o[:, step], self.lstmNode.uo)
        self.calculate_weights(num_steps, dim_corr_outp)
        return last_delta_h

    def calculate_weights(self, num_steps, dim_corr_outp):
        # calculate weight corrections
        if dim_corr_outp == 1:
            for n in reversed(range(num_steps)):
                self.delta_wg = self.delta_wg + (self.diff_state_g[n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wi = self.delta_wi + (self.diff_state_i[n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wo = self.delta_wo + (self.diff_state_o[n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wf = self.delta_wf + (self.diff_state_f[n] * self.lstmNode.x_list[n]).transpose()
            for n in reversed(range(num_steps - 1)):
                self.delta_ug = self.delta_ug + (self.diff_state_g[n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_ui = self.delta_ui + (self.diff_state_i[n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_uo = self.delta_uo + (self.diff_state_o[n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_uf = self.delta_uf + (self.diff_state_f[n + 1] * self.lstmNode.state_h_list[n]).transpose()
            for n in reversed(range(num_steps)):
                self.delta_bg = self.delta_bg + self.diff_state_g[n]
                self.delta_bi = self.delta_bi + self.diff_state_i[n]
                self.delta_bo = self.delta_bo + self.diff_state_o[n]
                self.delta_bf = self.delta_bf + self.diff_state_f[n]
        else:
            for n in reversed(range(num_steps)):
                self.delta_wg = self.delta_wg + (self.diff_state_g[:, n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wi = self.delta_wi + (self.diff_state_i[:, n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wo = self.delta_wo + (self.diff_state_o[:, n] * self.lstmNode.x_list[n]).transpose()
                self.delta_wf = self.delta_wf + (self.diff_state_f[:, n] * self.lstmNode.x_list[n]).transpose()
            for n in reversed(range(num_steps - 1)):
                self.delta_ug = self.delta_ug + (self.diff_state_g[:, n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_ui = self.delta_ui + (self.diff_state_i[:, n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_uo = self.delta_uo + (self.diff_state_o[:, n + 1] * self.lstmNode.state_h_list[n]).transpose()
                self.delta_uf = self.delta_uf + (self.diff_state_f[:, n + 1] * self.lstmNode.state_h_list[n]).transpose()
            for n in reversed(range(num_steps)):
                self.delta_bg = self.delta_bg + self.diff_state_g[:, n]
                self.delta_bi = self.delta_bi + self.diff_state_i[:, n]
                self.delta_bo = self.delta_bo + self.diff_state_o[:, n]
                self.delta_bf = self.delta_bf + self.diff_state_f[:, n]

        self.lstmNode.wg = self.lstmNode.wg - self.sgd_param * self.delta_wg
        self.lstmNode.wi = self.lstmNode.wi - self.sgd_param * self.delta_wi
        self.lstmNode.wo = self.lstmNode.wo - self.sgd_param * self.delta_wo
        self.lstmNode.wf = self.lstmNode.wf - self.sgd_param * self.delta_wf

        self.lstmNode.ug = self.lstmNode.ug - self.sgd_param * self.delta_ug
        self.lstmNode.ui = self.lstmNode.ui - self.sgd_param * self.delta_ui
        self.lstmNode.uo = self.lstmNode.uo - self.sgd_param * self.delta_uo
        self.lstmNode.uf = self.lstmNode.uf - self.sgd_param * self.delta_uf

        self.lstmNode.bg = self.lstmNode.bg - self.sgd_param * self.delta_bg
        self.lstmNode.bi = self.lstmNode.bi - self.sgd_param * self.delta_bi
        self.lstmNode.bo = self.lstmNode.bo - self.sgd_param * self.delta_bo
        self.lstmNode.bf = self.lstmNode.bf - self.sgd_param * self.delta_bf
