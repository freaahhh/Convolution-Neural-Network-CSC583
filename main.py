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

EPOCH = 100
BATCH_SIZE = 25
IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 4
SELECTED_CLASSES = ['angry', 'happy', 'sad', 'neutral']
BASE_DIR = r'CSC583\training'
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
        horizontal_flip=True,
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
    model.add(MaxPooling2D(2, 2))

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

def launch_gui(model, selected_classes,history,acc):

    
    def preprocess_image(image_path):
        img = Image.open(image_path).convert('L').resize((48, 48))
        img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
        return img_array
    
    def retrain_model():
        nonlocal model

        # Load and retrain
        base_dir = r'C:\Users\Ahmad\Python Workspace\CSC583\training'
        train_gen, val_gen, test_gen = load_data(base_dir, selected_classes)
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCH)
        model.save("expression_model.keras")
        with open("training_history.pkl", 'wb') as f:
            pickle.dump(history.history, f)
        
        loss, acc = model.evaluate(test_gen)
        acc_label.config(text=f"Current Accuracy : {acc:.2f}")

        result_label.config(text="Retraining completed!")


    def classify_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            predicted_class = selected_classes[np.argmax(prediction)]

            result_label.config(text=f"Predicted: {predicted_class}")

            # Show the image
            img = Image.open(file_path).resize((150, 150))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    # Create window
    root = tk.Tk()
    root.title("Facial Expression Detector")
    root.geometry("300x300")

    
    tk.Button(root, text="Choose Image", command=classify_image).pack(pady=10)
    tk.Button(root, text="Retrain Model", command=retrain_model).pack(pady=5)
    acc_label = tk.Label(root,text=f"Current Accuracy : {acc:.2f}")
    acc_label.pack()

    image_label = tk.Label(root)
    image_label.pack()
    result_label = tk.Label(root, text="", font=("Arial", 16))
    result_label.pack(pady=10)

    root.mainloop() #starts the gui (utk IDE buang ni pun takpe)

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
    base_dir = r'C:\Users\Ahmad\Python Workspace\CSC583\training'
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
