import tkinter as tk
from tkinter import ttk
from tkinter import *
import userInterfaceHelper as uih
import deepfake_userinterfacehelper as dp
from PIL import Image, ImageTk

LARGEFONT = ("Verdana", 35)
# FORGERY DETECTION USER INTERFACE -> FORGERY LOCALIZATION IN IMAGES DETECTION & DEEPFAKE DETECTION
class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, imageForgeryPage, deepFakePage):
            frame = F(container, self)

            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# first window frame startpage

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        middleframe = Frame(self, relief=tk.SOLID, borderwidth=2)
        middleframe.place(relx=0.5, rely=0.5, anchor=CENTER)

        img = Image.open("Forgery Detection.png")
        img = ImageTk.PhotoImage(img)
        labelBlocks = Label(middleframe, image=img)
        labelBlocks.image = img  # keep a reference! by attaching it to a widget attribute
        labelBlocks.grid()
        # labelBlocks.grid(row=0, column=2, padx=10, pady=10)

        buttonFrame = Frame(middleframe)
        buttonFrame.grid()

        button1 = ttk.Button(buttonFrame, text="Image Forgery Detection",
                             command=lambda: controller.show_frame(imageForgeryPage))
        button1.pack(side=LEFT, padx=15, pady=15)
        button2 = ttk.Button(buttonFrame, text="Deep Fake Detection",
                             command=lambda: controller.show_frame(deepFakePage))
        button2.pack(side=LEFT,padx=15, pady=15)


# IMAGE FORGERY DETECTION PAGE
class imageForgeryPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        button_back = ttk.Button(self, text="< Back to main page",
                             command=lambda: controller.show_frame(StartPage))

        # putting the button in its place by
        # using grid
        button_back.grid(row=0, column=0)

        label = ttk.Label(self, text="Image Forgery Detection", font=LARGEFONT)
        label.grid(row=0, column=1, padx=10, pady=10)

        leftframe = Frame(self)
        leftframe.grid(row=1, column=0, padx=5, pady=5)

        middleframe = Frame(self)
        middleframe.grid(row=1, column=1, padx=5, pady=5)

        rightframe = Frame(self)
        rightframe.grid(row=1, column=2, padx=5, pady=5)

        imageTamperedFrame = Frame(middleframe, bg="black")
        imageTamperedFrame.grid(row=1, column=1, columnspan=4, padx=5, pady=5)

        imageMFRFrame = Frame(middleframe, bg="black")
        imageMFRFrame.grid(row=1, column=2, columnspan=4, padx=5, pady=5)

        frame1 = Frame(leftframe)
        frame1.grid(row=1, sticky=tk.W)
        step1LBL = Label(frame1, text="Step 1: ")
        step1LBL.pack(side=LEFT)

        img = Image.open("placeholder.png")
        # Change the dimensions of the image first
        img_resized = img.resize((240, 240))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        # global labelTampered
        label = Label(middleframe, text="Uploaded Image")
        label.grid(row=0, column=1, padx=5, pady=5)
        labelTampered = Label(middleframe, image=img)
        labelTampered.image = img  # keep a reference! by attaching it to a widget attribute
        labelTampered.grid(row=1, column=1, padx=5, pady=5)

        uploadButton = Button(frame1, text='Upload Image', width=20, command=lambda: uih.upload_file(middleframe,labelTampered))
        uploadButton.pack(side=RIGHT, padx=3, pady=3)



        frame2 = Frame(leftframe)
        frame2.grid(row=2, sticky=tk.W)
        step2LBL = Label(frame2, text="Step 2: ")
        step2LBL.pack(side=LEFT)

        img = Image.open("placeholder.png")
        # Change the dimensions of the image first
        img_resized = img.resize((240, 240))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        # global labelTampered
        label = Label(middleframe, text="Median Filtered Residual")
        label.grid(row=3, column=1, padx=5, pady=5)
        labelMFR = Label(middleframe, image=img)
        labelMFR.image = img  # keep a reference! by attaching it to a widget attribute
        labelMFR.grid(row=4, column=1, padx=5, pady=5)
        MFRbutton = Button(frame2, text="Get MFR", command=lambda: uih.get_MFR(middleframe,labelMFR))
        MFRbutton.pack(side=RIGHT, padx=3, pady=3)

        frame3 = Frame(leftframe)
        frame3.grid(row=3, sticky=tk.W)
        step3LBL = Label(frame3, text="Step 3: Divide MFR into ")
        step3LBL.pack(side=LEFT)
        optionsDimensions = ["32 x 32 blocks", "64 x 64 blocks", "128 x 128 blocks"]
        clicked = StringVar()
        clicked.set("128 x 128 blocks")
        img = Image.open("placeholder.png")
        # Change the dimensions of the image first
        img_resized = img.resize((240, 240))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        # global labelTampered

        label_block = Label(rightframe, text="Divide into ? blocks")
        label_block.grid(row=0, column=2, padx=5, pady=5)

        labelBlocks = Label(rightframe, image=img)
        labelBlocks.image = img  # keep a reference! by attaching it to a widget attribute
        labelBlocks.grid(row=1, column=2, padx=5, pady=5)
        submitbutton = Button(frame3, text="Submit", command=lambda: uih.getImageBlocks(clicked, rightframe,label_block, labelBlocks))
        submitbutton.pack(side=RIGHT, padx=3, pady=3)
        drop = OptionMenu(frame3, clicked, *optionsDimensions)
        drop.pack(side=RIGHT, padx=3, pady=3)

        frame4 = Frame(leftframe)
        frame4.grid(row=4, sticky=tk.W)
        step4LBL = Label(frame4, text="Step 4: ")
        step4LBL.pack(side=LEFT)
        extractbutton = Button(frame4, text="Extract Features", command=lambda: uih.startExtraction(clicked,self))
        extractbutton.pack(side=RIGHT, padx=3, pady=3)

        frame5 = Frame(leftframe)
        frame5.grid(row=5, sticky=tk.W)
        step5LBL = Label(frame5, text="Step 5: Select ")
        step5LBL.pack(side=LEFT)
        optionsTampering = [
            "Median Filtering",
            "Contrast Enhancement",
            "Gaussian Filtering",
            "Wiener Filtering",
            "JPEG Compression",
            "Average Filtering",
            "Rescale Operation",
            "Additive White Gaussian Noise",
            "Unsharp Masking",
            "JPEG 2000"]
        clickedTampering = StringVar()
        clickedTampering.set("Median Filtering")
        dropTampering = OptionMenu(frame5, clickedTampering, *optionsTampering)
        dropTampering.pack(side=RIGHT, padx=3, pady=3)


        frame6 = Frame(leftframe)
        frame6.grid(row=6, sticky=tk.W)
        step5LBL = Label(frame6, text="Step 6: ")
        step5LBL.pack(side=LEFT)
        img = Image.open("placeholder.png")
        # Change the dimensions of the image first
        img_resized = img.resize((240, 240))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        # global labelTampered
        label = Label(rightframe, text="Forged regions highlighted in White")
        label.grid(row=2, column=2, padx=5, pady=5)

        labelRegions = Label(rightframe, image=img)
        labelRegions.image = img  # keep a reference! by attaching it to a widget attribute
        labelRegions.grid(row=3, column=2, columnspan=4, padx=5, pady=5)


        classifybutton = Button(frame6, text="Classify Forged Blocks",
                                command=lambda: uih.classifyForgedBlocks(clickedTampering, rightframe, clicked,labelRegions))
        classifybutton.pack(side=RIGHT, padx=3, pady=3)



# DEEPFAKE DETECTION PAGE
class deepFakePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        button_back = ttk.Button(self, text="< Back to main page",
                                 command=lambda: controller.show_frame(StartPage))

        # putting the button in its place by
        # using grid
        button_back.grid(row=0, column=0)
        label = ttk.Label(self, text="Deep Fake Detection", font=LARGEFONT)
        label.grid(row=0, column=1, padx=10, pady=10)

        leftframe = Frame(self)
        leftframe.grid(row=1, column=0, padx=5, pady=5)

        middleframe = Frame(self)
        middleframe.grid(row=1, column=1, padx=5, pady=5)

        rightframe = Frame(self)
        rightframe.grid(row=1, column=2, padx=5, pady=5)

        imageTamperedFrame = Frame(middleframe, bg="black")
        imageTamperedFrame.grid(row=1, column=1, columnspan=4, padx=5, pady=5)

        imageMFRFrame = Frame(middleframe, bg="black")
        imageMFRFrame.grid(row=1, column=2, columnspan=4, padx=5, pady=5)

        frame1 = Frame(leftframe)
        frame1.grid(row=1, sticky=tk.W)
        step1LBL = Label(frame1, text="Step 1: ")
        step1LBL.pack(side=LEFT)

        labelUpload = Label(middleframe, text="Uploaded Image")
        labelUpload.grid(row=0, column=2, padx=5, pady=5)

        img = Image.open("placeholder.png")
        # Change the dimensions of the image first
        img_resized = img.resize((240, 240))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        # global labelTampered
        imageUploadLabel = Label(middleframe, image=img)
        imageUploadLabel.image = img  # keep a reference! by attaching it to a widget attribute
        imageUploadLabel.grid(row=1, column=2, padx=5, pady=5)

        uploadButton = Button(frame1, text='Upload Image', width=20, command=lambda: dp.upload_file(middleframe,imageUploadLabel))
        uploadButton.pack(side=RIGHT, padx=3, pady=3)





        frame2 = Frame(leftframe)
        frame2.grid(row=2, sticky=tk.W)
        step2LBL = Label(frame2, text="Step 2: ")
        step2LBL.pack(side=LEFT)

        labelIdentify = Label(middleframe, text="Identified Eyes, Nose, Lips")
        labelIdentify.grid(row=3, column=2, padx=5, pady=5)

        imageIdentifyLabel = Label(middleframe, image=img)
        imageIdentifyLabel.image = img
        imageIdentifyLabel.grid(row=4, column=2, columnspan=4, padx=5, pady=5)


        identifybutton = Button(frame2, text="Identify eyes, nose, lips",
                                command=lambda: dp.identificationFacialParts(middleframe,imageIdentifyLabel))
        identifybutton.pack(side=RIGHT, padx=3, pady=3)

        frame3 = Frame(leftframe)
        frame3.grid(row=3, sticky=tk.W)
        step3LBL = Label(frame3, text="Step 3: ")
        step3LBL.pack(side=LEFT)
        MFRbutton = Button(frame3, text="Get MFR", command=lambda: dp.get_MFR())
        MFRbutton.pack(side=RIGHT, padx=3, pady=3)

        frame4 = Frame(leftframe)
        frame4.grid(row=4, sticky=tk.W)
        step4LBL = Label(frame4, text="Step 4: ")
        step4LBL.pack(side=LEFT)
        extractbutton = Button(frame4, text="Extract Features", command=lambda: dp.startExtraction(self))
        extractbutton.pack(side=RIGHT, padx=3, pady=3)

        frame6 = Frame(leftframe)
        frame6.grid(row=5, sticky=tk.W)
        step5LBL = Label(frame6, text="Step 5: ")
        step5LBL.pack(side=LEFT)

        label_pred = Label(middleframe, text="Predicted Class: ?", borderwidth=1, relief="solid")
        label_pred.grid(row=5, column=2, padx=5, pady=5)

        classifybutton = Button(frame6, text="Classify the Image",
                                command=lambda: dp.classifyImage(middleframe,label_pred))
        classifybutton.pack(side=RIGHT, padx=3, pady=3)



# Driver Code
app = tkinterApp()
app.title("Forgery Detection")
app.mainloop()