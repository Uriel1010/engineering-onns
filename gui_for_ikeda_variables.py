import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import io
import Ikeda_equation
import numpy as np
import inspect


class GUI:
    img = None

    def __init__(self, master):
        self.master = master
        self.entries = []
        self.variables = {'tau_d': 20.87E-6, 'Tr': 240e-9, 'beta': 0.3, 'mu': 2.5, 'phi_0': np.pi * 0.89, 'rho': 0.0}

        # create the function dropdown chooser
        u_functions = [name for name, func in inspect.getmembers(Ikeda_equation, inspect.isfunction) if
                       name.startswith('u')]
        self.func_var = tk.StringVar()
        self.func_dropdown = ttk.Combobox(self.master, textvariable=self.func_var, values=u_functions, state='readonly')
        self.func_dropdown.place(x=50, y=375)

        # Set the first item in the dropdown as the default choice
        if u_functions:
            self.func_dropdown.current(0)

        # create the canvas
        self.canvas = tk.Canvas(self.master, width=500, height=500, bg='#F0F0F0')
        self.canvas.grid(row=0, column=1, sticky='nsew')



        # configure grid
        self.master.grid_columnconfigure(0, weight=1, minsize=200)
        self.master.grid_columnconfigure(1, weight=3)
        self.master.grid_rowconfigure(0, weight=1)

        # create entries in the first column
        for i, var_name in enumerate(self.variables.keys()):
            label = tk.Label(self.master, text=var_name, bg='#F0F0F0')
            label.place(x=25, y=25+i*50)
            entry = tk.Entry(self.master)
            entry.insert(0, str(self.variables[var_name]))
            entry.place(x=75, y=25+i*50)
            self.entries.append(entry)

        # create the OK button
        self.ok_button = tk.Button(self.master, text="OK", command=self.ok_callback, bg='#4B8BBE', fg='white', padx=10, pady=5)
        self.ok_button.place(x=50, y=325)

        # initialize the image variable
        if GUI.img is None:
            GUI.img = tk.PhotoImage()

    def resize(self, event):
        if GUI.img:
            self.redraw_plot()

    def ok_callback(self):
        for i, (key, entry) in enumerate(zip(self.variables.keys(), self.entries)):
            try:
                self.variables[key] = float(entry.get())
            except ValueError:
                self.variables[key] = 0.0

        self.redraw_plot()

    def redraw_plot(self):
        # Add the chosen function to the plot
        chosen_func_name = self.func_var.get()
        if chosen_func_name:
            chosen_func = getattr(Ikeda_equation, chosen_func_name)
            # Call chosen_func with appropriate arguments and plot the result
            # You may need to modify this depending on the function signature

        sol = Ikeda_equation.initial_value_problem(self.variables['tau_d'],
                                                   self.variables['Tr'],
                                                   self.variables['beta'],
                                                   self.variables['mu'],
                                                   self.variables['phi_0'],
                                                   self.variables['rho'],
                                                   chosen_func)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(sol.t, sol.y[0])
        plt.xlabel('s')
        plt.ylabel('x')
        plt.title(
            'Reservoir Dynamics without External Signal\n'
            fr'($\beta$={self.variables["beta"]}, $\mu$={self.variables["mu"]}, $\Phi_0$={self.variables["phi_0"]:.2f})'
        )

        # instead of showing the plot, save it to a PNG image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_bytes = buf.getvalue()
        buf.close()

        # update the PhotoImage object with the new image data
        img = tk.PhotoImage(data=img_bytes)
        GUI.img = img

        # delete previous image if exists and create a new image object on the canvas
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=GUI.img,
                                 anchor='center')  # center the image


root = tk.Tk()
gui = GUI(root)
root.mainloop()
