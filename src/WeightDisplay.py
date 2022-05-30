from tkinter import *

class WeightDisplay:
    DEFAULT_COLOR = "#0000ff"

    def __init__(self, mainframe, nr_boxes, col_value, row_value):
        self.boxes = []
        box_dim = 20
        delta = 0
        self.canvas = Canvas(mainframe, width=40, height=25 * nr_boxes,)
        for cnt in range(0, nr_boxes):
            self.rect = self.canvas.create_rectangle(10, 10 + cnt * (box_dim + delta),
                                                     10 + box_dim, 10 + (cnt + 1) * box_dim + cnt * delta,
                                                     outline=self.DEFAULT_COLOR, fill=self.DEFAULT_COLOR)
            self.boxes.append(self.rect)
        self.canvas.grid(column=col_value, row=row_value, sticky="N S", padx = 25)

    def reset_color(self):
        for nr in range(0, len(self.boxes)):
            self.canvas.itemconfig(self.boxes[nr], fill=self.DEFAULT_COLOR)

    def set_value(self, box_nr, value):
        self.canvas.itemconfig(self.boxes[box_nr], fill=self.calc_color(value), outline=self.DEFAULT_COLOR)

    @staticmethod
    def calc_color(value):
        if 0.01 < value <= 0.25:
            return '#e5ffe5'
        elif 0.25 < value <= 0.5:
            return '#99ff99'
        elif 0.5 < value <= 0.75:
            return '#edff4d'
        elif 0.75 < value:
            return '#00ff00'
        elif -0.01 < value <= 0.01:
            return '#ffffff'
        elif -0.25 < value < -0.01:
            return '#ffe5e5'
        elif -0.5 < value <= -0.25:
            return '#ff9999'
        elif -0.75 < value <= -0.5:
            return '#ff4d4d'
        elif value <= -0.75:
            return '#ff0000'
