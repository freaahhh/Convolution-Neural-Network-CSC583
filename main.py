import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === Parameters ===
MODEL_PATH = "expression_model.keras"
HISTORY_PATH = "training_history.pkl"
BASE_DIR = r'C:\Users\Ahmad\Python Workspace\CSC583\training'
SELECTED_CLASSES = ['angry', 'happy', 'sad', 'neutral']


def load_data(base_dir):
    image_size = (48, 48)
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=image_size,
        color_mode='grayscale',
        classes=SELECTED_CLASSES,
        class_mode='categorical',
        batch_size=batch_size,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=image_size,
        color_mode='grayscale',
        classes=SELECTED_CLASSES,
        class_mode='categorical',
        batch_size=batch_size,
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=image_size,
        color_mode='grayscale',
        classes=SELECTED_CLASSES,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(3, 3),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(SELECTED_CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert('L').resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
    return img_array


def launch_gui(model, history, acc):
    def retrain():
        nonlocal model
        train_gen, val_gen, test_gen = load_data(BASE_DIR)
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=40)
        model.save(MODEL_PATH)
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        _, new_acc = model.evaluate(test_gen)
        acc_label.config(text=f"Current Accuracy : {new_acc:.2f}")
        result_label.config(text="Retraining complete!")

    def classify_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            img_array = preprocess_image(file_path)
            prediction = model.predict(img_array)
            predicted_idx = np.argmax(prediction)
            class_name = SELECTED_CLASSES[predicted_idx]
            confidence = prediction[0][predicted_idx] * 100

            result_label.config(text=f"{class_name} ({confidence:.2f}%)")

            img = Image.open(file_path).resize((150, 150))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    def test_all():
        test_dir = os.path.join(BASE_DIR, 'test')
        all_data = []

        for cls in SELECTED_CLASSES:
            path = os.path.join(test_dir, cls)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        all_data.append((os.path.join(path, file), cls))

        idx = 0
        totals = {cls: 0 for cls in SELECTED_CLASSES}
        counts = {cls: 0 for cls in SELECTED_CLASSES}

        def process():
            nonlocal idx
            if idx < len(all_data):
                path, true_class = all_data[idx]
                current_class_label.config(text=f"Current Class Testing: {true_class}")
                img_array = preprocess_image(path)
                prediction = model.predict(img_array)
                pred_idx = np.argmax(prediction)
                confidence = prediction[0][pred_idx] * 100
                totals[true_class] += confidence
                counts[true_class] += 1

                img = Image.open(path).resize((150, 150))
                img = ImageTk.PhotoImage(img)
                image_label.config(image=img)
                image_label.image = img
                result_label.config(text=f"{SELECTED_CLASSES[pred_idx]} ({confidence:.2f}%)")

                idx += 1
                root.after(10, process)
            else:
                avg = {cls: (totals[cls] / counts[cls] if counts[cls] else 0) for cls in SELECTED_CLASSES}
                plt.bar(avg.keys(), avg.values(), color='skyblue')
                plt.ylim(0, 100)
                plt.title("Average Confidence per Class")
                plt.ylabel("Confidence (%)")
                plt.tight_layout()
                plt.show()
                current_class_label.config(text="Finished all classes.")

        process()

    root = tk.Tk()
    root.title("Facial Expression Classifier")
    root.geometry("340x420")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    tk.Button(frame, text="Choose Image", command=classify_image).grid(row=0, column=0, padx=5)
    tk.Button(frame, text="Test All Classes", command=test_all).grid(row=0, column=1, padx=5)
    tk.Button(root, text="Retrain Model", command=retrain).pack(pady=5)

    acc_label = tk.Label(root, text=f"Current Accuracy : {acc:.2f}")
    acc_label.pack()

    current_class_label = tk.Label(root, text="Current Class Testing: None")
    current_class_label.pack()

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root, text="", font=("Arial", 14))
    result_label.pack(pady=10)

    root.mainloop()


def main():
    train_gen, val_gen, test_gen = load_data(BASE_DIR)
    if os.path.exists(MODEL_PATH) and os.path.exists(HISTORY_PATH):
        model = keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        _, acc = model.evaluate(test_gen)
        with open(HISTORY_PATH, 'rb') as f:
            history = pickle.load(f)
    else:
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=40)
        model.save(MODEL_PATH)
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        _, acc = model.evaluate(test_gen)

    launch_gui(model, history, acc)


if __name__ == "__main__":
    main()
