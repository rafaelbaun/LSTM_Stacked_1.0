import numpy as np

class LSTMNode:
    do_debug = False

    def __init__(self, x_dim, h_dim):
        # self.num_steps = num_steps
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.s_prev = 0.
        self.h_prev = np.zeros(self.h_dim)
        # self.x_matrix = None
        self.num_steps = 0
        self.x_list = []

        # states ###############################################################
        self.state_g = np.zeros(self.h_dim)
        self.state_i = np.zeros(self.h_dim)
        self.state_f = np.zeros(self.h_dim)
        self.state_o = np.zeros(self.h_dim)
        self.state_s = np.zeros(self.h_dim)
        self.state_h = np.zeros(self.h_dim)

        #state for input activation
        self.tanh_state_h = np.zeros(self.h_dim)

        # store states per step/iteration of epoch
        self.state_g_list = []
        self.state_i_list = []
        self.state_f_list = []
        self.state_o_list = []
        self.state_s_list = []
        self.state_h_list = []
        self.state_h_list = []
        self.tanh_state_h_list = []

        # weight matrices ######################################################
        # weights of input gate
        self.wg = self.rand_arr(-0.1, 0.1, self.h_dim, self.x_dim)
        self.ug = self.rand_arr(-0.1, 0.1, self.h_dim, self.h_dim)
        self.wi = self.rand_arr(-0.1, 0.1, self.h_dim, self.x_dim)
        self.ui = self.rand_arr(-0.1, 0.1, self.h_dim, self.h_dim)
        # weights of forget gate
        self.wf = self.rand_arr(-0.1, 0.1, self.h_dim, self.x_dim)
        self.uf = self.rand_arr(-0.1, 0.1, self.h_dim, self.h_dim)
        # weights of output gate
        self.wo = self.rand_arr(-0.1, 0.1, self.h_dim, self.x_dim)
        self.uo = self.rand_arr(-0.1, 0.1, self.h_dim, self.h_dim)

        # bias terms ###########################################################
        # bias of input gate
        self.bg = self.rand_arr(-0.1, 0.1, self.h_dim)
        self.bi = self.rand_arr(-0.1, 0.1, self.h_dim)
        # bias of forget gate
        self.bf = self.rand_arr(-0.1, 0.1, self.h_dim)
        # bias of output gate
        self.bo = self.rand_arr(-0.1, 0.1, self.h_dim)

    # create uniform random array of values in [a,b) and shape args
    def rand_arr(self, a, b, *args):
        np.random.seed(0)
        return np.random.rand(*args) * (b - a) + a

    def sigmoid(self, z) -> float:
        return 1. / (1 + np.exp((-1)*z))

    def debug(self, msg):
        if self.do_debug:
            print(msg)

    # x(t) ... x ... token at t
    # h(t-1) ... h_prev ... hidden state at t-1
    # s(t-1) ... s_prev ... cell state at t-1
    def calculate(self, x, s_prev = None, h_prev = None):
        if s_prev is not None:
            self.s_prev = s_prev
        if h_prev is not None:
            self.h_prev = h_prev

        self.x_list.append(x)
        # INPUT GATE
        self.state_g = np.tanh(np.dot(self.wg, x) + np.dot(self.ug, self.h_prev) + self.bg)
        self.state_i = self.sigmoid(self.wi.dot(x) + self.ui.dot(self.h_prev) + self.bi)
        self.debug("state_g:")
        self.debug(self.state_g)
        self.debug("state_i:")
        self.debug(self.state_i)
        # FORGET GATE
        self.state_f = self.sigmoid(self.wf.dot(x) + self.uf.dot(self.h_prev) + self.bf)
        self.debug("state_f:")
        self.debug(self.state_f)
        # OUTPUT GATE
        self.state_o = self.sigmoid(self.wo.dot(x) + self.uo.dot(self.h_prev) + self.bo)
        self.debug("state_o:")
        self.debug(self.state_o)
        # Output
        self.s_prev = self.state_g * self.state_i + self.s_prev * self.state_f
        self.state_s = self.s_prev
        self.h_prev = np.tanh(self.state_s) * self.state_o
        self.state_h = self.h_prev

        #calculation for input activation
        self.tanh_state_h = np.tanh(self.h_prev)


        # store this iteration of the epoch
        self.state_g_list.append(self.state_g)
        self.state_i_list.append(self.state_i)
        self.state_f_list.append(self.state_f)
        self.state_o_list.append(self.state_o)
        self.state_s_list.append(self.state_s)
        self.state_h_list.append(self.state_h)
        self.tanh_state_h_list.append(self.tanh_state_h)

        self.num_steps += 1 # incease index of iteration of epoch
        return [self.state_s, self.state_h]

    # reset of state list; used when starting new epoch
    def reset_state_lists(self):
        self.num_steps = 0
        self.state_g_list = []
        self.state_i_list = []
        self.state_f_list = []
        self.state_o_list = []
        self.state_s_list = []
        self.state_h_list = []
        self.tanh_state_h_list = []

    def store(self, file_name, data):
        with open(file_name, 'a') as f:
            np.savetxt(f, data, delimiter=',')

    def store_all(self, dir, prefix):
        self.store(dir + prefix + '_wg.csv', self.wg)
        self.store(dir + prefix + '_wi.csv', self.wi)
        self.store(dir + prefix + '_wf.csv', self.wf)
        self.store(dir + prefix + '_wo.csv', self.wo)

        self.store(dir + prefix + '_ug.csv', self.ug)
        self.store(dir + prefix + '_ui.csv', self.ui)
        self.store(dir + prefix + '_uf.csv', self.uf)
        self.store(dir + prefix + '_uo.csv', self.uo)

        self.store(dir + prefix + '_bg.csv', self.bg)
        self.store(dir + prefix + '_bi.csv', self.bi)
        self.store(dir + prefix + '_bf.csv', self.bf)
        self.store(dir + prefix + '_bo.csv', self.bo)

        self.store(dir + prefix + '_tanh_state_h.csv', self.tanh_state_h)

    def store_gates(self, dir, prefix):
        self.store(dir + prefix + '_gate_g.csv', self.state_g)
        self.store(dir + prefix + '_gate_i.csv', self.state_i)
        self.store(dir + prefix + '_gate_f.csv', self.state_f)
        self.store(dir + prefix + '_gate_o.csv', self.state_o)
