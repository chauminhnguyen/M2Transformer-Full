from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import pipeline


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Python Tkinter Dialog Widget")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text="Open File")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)
        self.labelFrame2 = ttk.LabelFrame(self, text="Caption")
        self.labelFrame2.grid(column=1, row=1, padx=50, pady=20)
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.caplabel = ttk.Label(self.labelFrame2, text="")
        self.caplabel.grid(column=0, row=0)
        self.imglabel = ttk.Label(self.labelFrame2, image=None)
        self.imglabel.grid(column=0, row=1)
        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text="Browse A File", command=self.fileDialog)
        self.button.grid(column=1, row=1)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select A File",
            filetype=(("jpeg files", "*.jpg"), ("all files", "*.*"))
        )
        self.label.configure(text=self.filename)

        img = Image.open(self.filename)
        img = img.resize((450, 350), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        self.imglabel.configure(image=photo)
        self.imglabel.image = photo

        cap = pipeline.main(self.filename)
        cap = ' '.join(cap[0])
        self.caplabel.configure(text=cap)


root = Root()
root.mainloop()
