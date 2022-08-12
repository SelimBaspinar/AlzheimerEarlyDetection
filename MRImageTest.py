#SELİM BAŞPINAR - 44128127874
#Mehmet Ali BALİ - 63925397150
#Mustafa BÖLÜK - 48301492506

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                            filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 150  # Processing image for dysplaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text=str(
        file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    original = Image.open(image_data).convert("RGB")
    original = original.resize((128, 128))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    CATEGORIES = ['MildDemented', 'ModerateDemented',
                  'NonDemented', 'VeryMildDemented']  # will use this to convert prediction num to string value
    CATEGORIES_np = np.array(['CATEGORIES'])
    datagen_test = ImageDataGenerator(
        # image preprocessing function
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    )
    model = tf.keras.models.load_model(
        "./mri_resnet50.h5")

    predictions = model.predict(datagen_test.flow(
        image_batch))

    preds = np.argmax(np.array(predictions), axis=1)
    arr = str(predictions[0]).replace("[", "").replace("]", "").split(" ")
    print(CATEGORIES[preds[0]])
    table = tk.Label(
        frame, text=("Classified as:  "+CATEGORIES[preds[0]]), fg="red").pack()
    count = 0
    for i in range(len(arr)):

        if arr[i] != "":
            if count == 0:
                table = tk.Label(
                    frame, text="MildDemented" + "%" + str(round(float(arr[i])*100, 2))).pack()
            if count == 1:
                table = tk.Label(
                    frame, text="ModerateDemented" + "%" + str(round(float(arr[i])*100, 2))).pack()
            if count == 2:
                table = tk.Label(
                    frame, text="NonDemented" + "%" + str(round(float(arr[i])*100, 2))).pack()
            if count == 3:
                table = tk.Label(
                    frame, text="VeryMildDemented" + "%" + str(round(float(arr[i])*100, 2))).pack()
            count += 1

    table = tk.Label(
        frame, text=(CATEGORIES[preds[0]]), bg="red")


root = tk.Tk()
root.title('Portable Image Classifier')
root.resizable(False, False)
tit = tk.Label(root, text="Portable Image Classifier",
               padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=25, pady=5,
                        fg="white", bg="grey", command=load_img)

chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=25, pady=5,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

root.mainloop()