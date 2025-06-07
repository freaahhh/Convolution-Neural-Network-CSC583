import os
from tensorflow import keras
from keras import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Python code implementing the 
# image recognition/classification program, including 
# data preprocessing, model building, training, evaluation and GUI.<---------------------------------

# # Test data (no augmentation)
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     os.path.join(base_dir, 'test'),
#     target_size=(48, 48),
#     color_mode='grayscale',
#     classes=selected_classes,
#     class_mode='categorical',
#     batch_size=64,
#     shuffle=False
# )

# ====================
# === CONFIGURATION == nak tambah apa-apa kat sini ja 
# ====================

BASE_DIR = r'D:\Xampp\htdocs\Convolution-Neural-Network-CSC583\training' # <================================= ubah ni dulu

EPOCH = 60
BATCH_SIZE = 25
IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 4
SELECTED_CLASSES = ['angry', 'happy', 'sad', 'neutral']
MODEL_PATH = "expression_model.keras"
HISTORY_PATH = "training_history.pkl"

LAYER_CONFIG = [
    {"filters": 32, "kernel_size": (3, 3)},
    {"filters": 64, "kernel_size": (3, 3)},
    # {"filters": 64,(SIZE FILTER) "kernel_size": (3, 3)(MATRIX SIZE)},
    # Tambah layer kat sini kalau nak testing ikut kat atas tu
]

DENSE_UNITS = 128
DROPOUT_RATE = 0.5


#LOAD DATA INTO MODEL

def load_data(base_dir, selected_classes):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_model():
    model = Sequential()
    model.add(Conv2D(LAYER_CONFIG[0]["filters"], LAYER_CONFIG[0]["kernel_size"], activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(3, 3))

    model.add(Conv2D(LAYER_CONFIG[1]["filters"], LAYER_CONFIG[1]["kernel_size"], activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # Optionally add more Conv2D + MaxPool2D layers here from LAYER_CONFIG

    model.add(Flatten())
    model.add(Dense(DENSE_UNITS, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#GUI HEREEEEEEEEEEEEEEEEEEEE

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle

import matplotlib.pyplot as plt

def launch_gui(model, selected_classes, history, acc):
    def preprocess_image(image_path):
        img = Image.open(image_path).convert('L').resize((48, 48))
        img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
        return img_array

    def retrain_model():
        nonlocal model
        train_gen, val_gen, test_gen = load_data(BASE_DIR, selected_classes)
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCH)
        model.save(MODEL_PATH)
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)

        loss, new_acc = model.evaluate(test_gen)
        acc_label.config(text=f"Current Accuracy : {new_acc:.2f}")
        result_label.config(text="Retraining completed!")

    def classify_single_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            folder_name = os.path.basename(os.path.dirname(file_path))
            current_class_label.config(text=f"Current Folder Testing: {folder_name}")


            predicted_index = np.argmax(prediction)
            predicted_class = selected_classes[predicted_index]
            confidence = prediction[0][predicted_index] * 100

            result_label.config(text=f"{predicted_class} ({confidence:.2f}%)")

            img = Image.open(file_path).resize((150, 150))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    def classify_folder():
        folder_path = filedialog.askdirectory(title="Select Folder of Images")
        if folder_path:
            confidences = []
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            current_class_label.config(text=f"Current Folder Testing: {os.path.basename(folder_path)}")

            def process_image(index):
                if index < len(files):
                    img_path = os.path.join(folder_path, files[index])
                    img_array = preprocess_image(img_path)
                    prediction = model.predict(img_array)
                    confidence = np.max(prediction) * 100
                    predicted_class = selected_classes[np.argmax(prediction)]
                    confidences.append(confidence)

                    img = Image.open(img_path).resize((150, 150))
                    img = ImageTk.PhotoImage(img)
                    image_label.config(image=img)
                    image_label.image = img
                    result_label.config(text=f"{predicted_class} ({confidence:.2f}%) [{index+1}/{len(files)}]")

                    root.after(10, lambda: process_image(index + 1))
                else:
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    result_label.config(text=f"Confidence: {avg_conf:.2f}% on {len(confidences)} images")

            process_image(0)

    def test_all_classes_graph():
        test_dir = os.path.join(BASE_DIR, 'test')
        class_confidences = {}
        all_image_paths = []

        # Prepare all images with labels
        for expression in selected_classes:
            expression_path = os.path.join(test_dir, expression)
            if not os.path.exists(expression_path):
                continue

            image_files = [
                os.path.join(expression_path, file)
                for file in os.listdir(expression_path)
                if file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            all_image_paths.extend([(img_path, expression) for img_path in image_files])

        index = 0
        totals = {cls: 0 for cls in selected_classes}
        counts = {cls: 0 for cls in selected_classes}

        def process_next():
            nonlocal index
            if index < len(all_image_paths):
                img_path, true_class = all_image_paths[index]
                current_class_label.config(text=f"Current Class Testing: {true_class.capitalize()}")

                img_array = preprocess_image(img_path)
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = selected_classes[predicted_class_index]
                confidence = prediction[0][predicted_class_index] * 100

                totals[true_class] += confidence
                counts[true_class] += 1

                # Display image
                img = Image.open(img_path).resize((150, 150))
                img = ImageTk.PhotoImage(img)
                image_label.config(image=img)
                image_label.image = img

                result_label.config(text=f"{predicted_class} ({confidence:.2f}%) [{index+1}/{len(all_image_paths)}]")

                index += 1
                root.after(10, process_next)  # Delay in ms before processing next image
            else:
                # Done with all images — show bar chart
                for cls in selected_classes:
                    if counts[cls]:
                        class_confidences[cls] = totals[cls] / counts[cls]
                    else:
                        class_confidences[cls] = 0

                # Define a list of colors for each bar (match length to number of classes)
                colors = ["#88261b", "#f5bf0c", "#0d4569", "#106E37"] 

                plt.figure(figsize=(8, 5), facecolor="#e8d7f7")
                ax = plt.gca()  # Get current axes
                ax.set_facecolor("#f5f1cf")  # Inner plot area background
                plt.bar(class_confidences.keys(), class_confidences.values(), color=colors)
                plt.xlabel("Facial Expression Class")
                plt.ylabel("Average Confidence (%)")
                plt.title("Model Confidence per Class (Test Set)")
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.show()

                current_class_label.config(text="Finished all classes.")

        process_next()

    # GUI
    root = tk.Tk()
    root.title("Facial Expression Detector")
    root.geometry("340x420")
    root.configure(bg="#e8d7f7") #Light purple background

    btn_frame = tk.Frame(root, bg="#e8d7f7")
    btn_frame.pack(pady=10)

    # Base button style (shared)
    base_button_style = {
        "fg": "white",
        "activeforeground": "white",
        "font": ("Arial", 10, "bold"),
        "width": 15
    }

    # Individual styles using base + overrides
    button1_style = base_button_style.copy()
    button1_style.update({
        "bg": "#A020F0",
        "activebackground": "#F58461"
    })

    button2_style = base_button_style.copy()
    button2_style.update({
        "bg": "#3C0470",
        "activebackground": "#F58461"
    })

    button3_style = base_button_style.copy()
    button3_style.update({
        "bg": "#5B3B6B",
        "activebackground": "#F58461",
        "activeforeground": "black"  # override one value if needed
    })

    # RESULT at the top
    result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e8d7f7", fg="#3C0470")
    result_label.pack(pady=10)

    # IMAGE
    image_label = tk.Label(root, bg="#e8d7f7")  # No image yet
    image_label.pack()

    # ACCURACY + CURRENT CLASS
    acc_label = tk.Label(root, text=f"Current Accuracy : {acc:.2f}", bg="#e8d7f7", fg="#340738", font=("Arial", 10, "bold"))
    acc_label.pack()

    current_class_label = tk.Label(root, text="Current Class Testing: None", bg="#e8d7f7", fg="#340738", font=("Arial", 10, "bold"))
    current_class_label.pack()

    # BUTTONS
    btn_frame = tk.Frame(root, bg="#e8d7f7")
    btn_frame.pack(pady=5)

    tk.Button(btn_frame, text="Choose Image", command=classify_single_image, **button1_style).grid(row=0, column=0, padx=5)
    tk.Button(btn_frame, text="Choose Folder", command=classify_folder, **button1_style).grid(row=0, column=1, padx=5)

    tk.Button(root, text="Test All Classes", command=test_all_classes_graph, **button2_style).pack(pady=5)
    tk.Button(root, text="Retrain Model", command=retrain_model, **button3_style).pack(pady=5)

    root.mainloop()

# # graph plotting
# import matplotlib.pyplot as plt

# def plot_training_progress(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs_range = range(len(acc))

#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Accuracy Over Epochs')

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Training Loss')
#     plt.plot(epochs_range, val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.title('Loss Over Epochs')

#     plt.tight_layout()
#     plt.show()



def main():
    base_dir = BASE_DIR
    selected_classes = ['angry', 'happy', 'sad', 'neutral']
    acc = 0
    model_path = "expression_model.keras"
    history_path = "training_history.pkl"

    train_gen, val_gen, test_gen = load_data(base_dir, selected_classes)

    # If model and history exist
    if os.path.exists(model_path) and os.path.exists(history_path):
        print("Loading saved model and training history...")
        model = keras.models.load_model(model_path)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile

        loss, acc = model.evaluate(test_gen)  # Now test_gen is already loaded
        print(f"Test accuracy: {acc:.2f}")

        with open(history_path, 'rb') as f:
            history = pickle.load(f)

    else:
        # Train from scratch
        model = build_model()
        model.summary()

        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCH)

        # Save model and history
        model.save(model_path)
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

        # Evaluate
        loss, acc = model.evaluate(test_gen)
        print(f"Test accuracy: {acc:.2f}")

        

    # Optional: visualize training history
    # plot_training_progress(history)

    # Launch GUI
    launch_gui(model, selected_classes, history,acc)



# # MAIN LOGIC
# main testing 
# def main(): 
#     base_dir = r'C:\Users\Ahmad\Python Workspace\CSC583\training'  #TEMP
#     selected_classes = ['angry', 'happy', 'sad', 'neutral']

#     train_gen, val_gen, test_gen = load_data(base_dir, selected_classes)

#     model = build_model()
#     model.summary()

#     model.fit(train_gen, validation_data=val_gen, epochs=100)

#     loss, acc = model.evaluate(test_gen) #test the 5 image
#     print(f"Test accuracy: {acc:.2f}")
    

if __name__ == "__main__":
    main()
