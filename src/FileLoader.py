import csv


class FileLoader:
    bf_data = []
    bg_data = []
    bi_data = []
    bo_data = []
    uf_data = []
    ug_data = []
    ui_data = []
    uo_data = []
    wf_data = []
    wg_data = []
    wi_data = []
    wo_data = []
    gate_g_data = []
    gate_i_data = []
    gate_o_data = []
    gate_f_data = []

    def __init__(self, data_dir, prefix):
        self.prefix = prefix
        self.data_dir = data_dir

    def load_scalar_data(self, scalar_name, scalar_data):
        with open(self.data_dir + self.prefix + "_" + scalar_name + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                scalar_data.append(float(row[0]))

    def load_all_scalars(self):
        self.load_scalar_data("bf", self.bf_data)
        self.load_scalar_data("bg", self.bg_data)
        self.load_scalar_data("bi", self.bi_data)
        self.load_scalar_data("bo", self.bo_data)
        self.load_scalar_data("uf", self.uf_data)
        self.load_scalar_data("ug", self.ug_data)
        self.load_scalar_data("ui", self.ui_data)
        self.load_scalar_data("uo", self.uo_data)
        self.load_scalar_data("gate_g", self.gate_g_data)
        self.load_scalar_data("gate_i", self.gate_i_data)
        self.load_scalar_data("gate_f", self.gate_f_data)
        self.load_scalar_data("gate_o", self.gate_o_data)

    def load_vector_data(self, vector_name, vector_data):
        with open(self.data_dir + self.prefix + "_" + vector_name + ".csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                matrix_elems = []
                for part in row:
                    matrix_elems.append(float(part))  # convert matrix elements to floats
                vector_data.append(matrix_elems)

    def load_all_vectors(self):
        self.load_vector_data("wf", self.wf_data)
        self.load_vector_data("wg", self.wg_data)
        self.load_vector_data("wi", self.wi_data)
        self.load_vector_data("wo", self.wo_data)


# Testing ...
if __name__ == '__main__':
    fileLoader = FileLoader("../data/sinus/", "lstm_sinus_simple")
    fileLoader.load_all_scalars()
    fileLoader.load_all_vectors()
    print(fileLoader.bf_data)
    print(fileLoader.uf_data)
    print(fileLoader.wf_data)