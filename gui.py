from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import pipeline


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Image Captioning")
        self.minsize(640, 500)
        self.maxsize(640, 500)

        self.labelFrame = ttk.LabelFrame(self, text="Open File", borderwidth=0)
        # self.labelFrame.grid(column=0, row=2, padx=20, pady=10)
        # self.labelFrame.grid(column=0, row=0, sticky="ew")
        self.labelFrame.place(x=20, y=0, width=600, height=100)
        self.labelFrame2 = ttk.LabelFrame(self, text="Caption", borderwidth=0)
        # self.labelFrame2.grid(column=0, row=3, padx=20, pady=10)
        self.labelFrame2.place(x=20, y=100, width=600, height=400)

        self.label = ttk.Label(self.labelFrame, text="")
        self.label.pack(expand=1, fill='both')
        self.label.grid(column=0, row=0)

        self.caplabel = ttk.Label(self.labelFrame2, text="")
        self.caplabel.grid(column=0, row=0)

        self.imglabel = ttk.Label(self.labelFrame2, image=None)
        self.imglabel.grid(column=0, row=1)
        self.button()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text="Browse A File", command=self.fileDialog)
        self.button.grid(column=0, row=0)
        self.button.place(x=300, y=20, anchor="center")

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select A File",
            filetype=(("jpeg files", "*.jpg"), ("all files", "*.*"))
        )

        if self.filename != "":
            self.label.configure(text=self.filename)
            self.label.place(x=300, y=40, anchor='center')

            img = Image.open(self.filename)
            img = img.resize((450, 350), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(img)
            self.imglabel.configure(image=photo)
            self.imglabel.image = photo
            self.imglabel.place(x=300, y=200, anchor='center')

            cap = pipeline.main(self.filename)
            cap = ' '.join(cap[0])
            self.caplabel.configure(text=cap)
            self.caplabel.place(x=300, y=10, anchor='center')


root = Root()
root.mainloop()
