from tkinter import ttk
from tkinter import *
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import ImageTk, Image
from WeightDisplay import WeightDisplay
from FileLoader import FileLoader


class LSTMMonitor:
    img = None
    step = 0
    nr_epoch = 0
    nr_test_wdw = 21

    def select_sim(self, event):
        if self.value_inside.get() == "Automatisch":
            self.start_button.state(["!disabled"])
            self.step_button.state(["disabled"])
        elif self.value_inside.get() == "Schrittweise":
            self.start_button.state(["disabled"])
            self.step_button.state(["!disabled"])

    def run_a_step(self):
        for n in range(0, 5):
            self.weightDispWG.set_value(n, self.fileLoader.wg_data[self.step][n])
        for n in range(0, 5):
            self.weightDispWI.set_value(n, self.fileLoader.wi_data[self.step][n])
        for n in range(0, 5):
            self.weightDispWF.set_value(n, self.fileLoader.wf_data[self.step][n])
        for n in range(0, 5):
            self.weightDispWO.set_value(n, self.fileLoader.wo_data[self.step][n])
        self.weightDispUG.set_value(0, self.fileLoader.ug_data[self.step])
        self.weightDispUF.set_value(0, self.fileLoader.uf_data[self.step])
        self.weightDispUI.set_value(0, self.fileLoader.ui_data[self.step])
        self.weightDispUO.set_value(0, self.fileLoader.uo_data[self.step])

    def automatic_run(self):
        if self.step < self.nr_epoch:
            self.run_a_step()
            self.start_button.configure(text="Starten " + str(self.step + 1) + "/" + str(self.nr_epoch))
            self.step += 1
            self.input_labelFrame.after(100, self.automatic_run)

    def stepwise_run(self):
        if self.step < self.nr_epoch:
            self.run_a_step()
            self.step_button.configure(text="Schritt " + str(self.step + 1) + "/" + str(self.nr_epoch))
            self.step += 1

    def reset(self):
        self.step = 0
        self.step_button.state(["disabled"])
        self.step_button.configure(text="Schritt")
        self.start_button.state(["disabled"])
        self.start_button.configure(text="Starten")
        self.value_inside.set("Auswählen")
        self.weightDispWG.reset_color()
        self.weightDispWF.reset_color()
        self.weightDispWI.reset_color()
        self.weightDispWO.reset_color()
        self.weightDispUF.reset_color()
        self.weightDispUG.reset_color()
        self.weightDispUI.reset_color()
        self.weightDispUO.reset_color()

    def __init__(self, root_pane, nr_epoch):
        self.nr_epoch = nr_epoch
        self.fileLoader = FileLoader("../data/sinus/", "lstm_sinus_simple")
        self.fileLoader.load_all_scalars() # loads also gate data for testing
        self.fileLoader.load_all_vectors()

        root_pane.title("LSTM Simulator")
        mainframe = ttk.Frame(root_pane, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky="N, W, E, S")
        root_pane.columnconfigure(0, weight=1)
        root_pane.rowconfigure(0, weight=1)

        img = Image.open("LSTM.png")
        img = img.resize((1000, 450))
        self.photoImg = ImageTk.PhotoImage(img)
        panel = ttk.Label(mainframe, image=self.photoImg)
        panel.image = self.photoImg
        panel.grid(column=1, row=0, columnspan=6, rowspan=2, sticky=N)

        Label(mainframe, text="TRAINING", font=("Arial", 18)).grid(column=0, row=2, columnspan=6, sticky=W)

        self.input_labelFrame = ttk.LabelFrame(mainframe, text='Input')
        self.input_labelFrame.grid(column=0, row=3, columnspan=5, sticky=W)

        forget_label_frame = ttk.LabelFrame(mainframe, text='Forget')
        forget_label_frame.grid(column=5, row=3, columnspan=1, sticky=W)

        output_label_frame = ttk.LabelFrame(mainframe, text='Output')
        output_label_frame.grid(column=6, row=3, columnspan=1, sticky=W)

        nr_boxes = 5
        Label(self.input_labelFrame, text="wg").grid(column=0, row=0)
        self.weightDispWG = WeightDisplay(self.input_labelFrame, nr_boxes, 0, 1)
        Label(self.input_labelFrame, text="ug").grid(column=1, row=0)
        self.weightDispUG = WeightDisplay(self.input_labelFrame, 1, 1, 1)
        Label(self.input_labelFrame, text="wi").grid(column=2, row=0)
        self.weightDispWI = WeightDisplay(self.input_labelFrame, nr_boxes, 2, 1)
        Label(self.input_labelFrame, text="ui").grid(column=3, row=0)
        self.weightDispUI = WeightDisplay(self.input_labelFrame, 1, 3, 1)

        Label(forget_label_frame, text="wf").grid(column=0, row=0)
        self.weightDispWF = WeightDisplay(forget_label_frame, nr_boxes, 0, 1)
        Label(forget_label_frame, text="uf").grid(column=1, row=0)
        self.weightDispUF = WeightDisplay(forget_label_frame, 1, 1, 1)

        Label(output_label_frame, text="wo").grid(column=0, row=0)
        self.weightDispWO = WeightDisplay(output_label_frame, nr_boxes, 0, 1)
        Label(output_label_frame, text="uo").grid(column=1, row=0)
        self.weightDispUO = WeightDisplay(output_label_frame, 1, 1, 1)

        self.sim_labelFrame = ttk.LabelFrame(mainframe, text='Simulation')
        self.sim_labelFrame.grid(column=7, row=3, columnspan=2, sticky=N, padx=10)

        self.value_inside: StringVar = StringVar(self.sim_labelFrame)
        self.value_inside.set("Auswählen")
        options_list: list[str] = ["Auswählen", "Automatisch", "Schrittweise"]
        self.sim_menu = ttk.OptionMenu(self.sim_labelFrame, self.value_inside, *options_list, command=self.select_sim)
        self.sim_menu.pack()
        self.sim_menu.grid(column=0, row=0, columnspan=1, sticky=W, pady=3)

        self.start_button = ttk.Button(self.sim_labelFrame, text="Starten", width=20, state="disabled",
                                       command=self.automatic_run)
        self.start_button.grid(column=0, row=1, sticky=W, pady=3)

        self.step_button = ttk.Button(self.sim_labelFrame, text="Schritt", width=20, state="disabled",
                                      command=self.stepwise_run)
        self.step_button.grid(column=0, row=2, sticky=W, pady=3)

        self.reset_button = ttk.Button(self.sim_labelFrame, text="Reset", width=20, command=self.reset)
        self.reset_button.grid(column=0, row=3, sticky=W, pady=3)

        # testing part
        Label(mainframe, text="TESTLAUF", font=("Arial", 18)).grid(column=0, row=4, columnspan=6, sticky=W)
        self.input_gate_labelFrame = ttk.LabelFrame(mainframe, text='Input')
        self.input_gate_labelFrame.grid(column=0, row=5, columnspan=5, sticky=W, pady = 15)

        self.input_i_gate_labelFrame = ttk.LabelFrame(self.input_gate_labelFrame, text='I - Gate')
        self.input_i_gate_labelFrame.grid(column=0, row=0, columnspan=1, sticky=W, padx = 5, pady=5)
        self.show_diag(self.fileLoader.gate_i_data, self.input_i_gate_labelFrame)

        self.input_g_gate_labelFrame = ttk.LabelFrame(self.input_gate_labelFrame, text='G - Gate')
        self.input_g_gate_labelFrame.grid(column=1, row=0, columnspan=1, sticky=W, padx = 5, pady=5)
        self.show_diag(self.fileLoader.gate_g_data, self.input_g_gate_labelFrame)

        self.forget_gate_label_frame = ttk.LabelFrame(mainframe, text='Forget')
        self.forget_gate_label_frame.grid(column=5, row=5, columnspan=1, sticky=W, pady=15)
        self.forget_f_gate_labelFrame = ttk.LabelFrame(self.forget_gate_label_frame, text='F - Gate')
        self.forget_f_gate_labelFrame.grid(column=0, row=0, columnspan=1, sticky=W, padx = 5, pady=5)
        self.show_diag(self.fileLoader.gate_f_data, self.forget_f_gate_labelFrame)

        self.output_gate_label_frame = ttk.LabelFrame(mainframe, text='Output')
        self.output_gate_label_frame.grid(column=6, row=5, columnspan=1, sticky=W, pady=15)
        self.output_o_gate_labelFrame = ttk.LabelFrame(self.output_gate_label_frame, text='O - Gate')
        self.output_o_gate_labelFrame.grid(column=0, row=0, columnspan=1, sticky=W, padx = 5, pady=5)
        self.show_diag(self.fileLoader.gate_o_data, self.output_o_gate_labelFrame)

    def show_diag(self, data, pane):
        fig_g = Figure(figsize=(2, 1), dpi=100)
        plot_g = fig_g.add_subplot(111)
        plot_g.plot(data)
        canvas = FigureCanvasTkAgg(fig_g, master=pane)
        fig_g.tight_layout(pad=0)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


if __name__ == '__main__':
    root = Tk()
    LSTMMonitor(root, 3) # nr_epoch = 3
    root.mainloop()
