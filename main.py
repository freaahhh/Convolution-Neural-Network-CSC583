import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pickle
import matplotlib.pyplot as plt

TF_USE_LEGACY_KERAS=True
from tensorflow import keras
from keras import Sequential
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.regularizers import l2


# === Parameters ===
MODEL_PATH = "expression_model.keras"
HISTORY_PATH = "training_history.pkl"
BASE_DIR = r'C:\Users\Ahmad\Python Workspace\CSC583\training' # Change this to your dataset path
SELECTED_CLASSES = ['angry', 'happy', 'sad', 'neutral']


def load_data(base_dir):
    image_size = (48, 48)
    batch_size = 128

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


def build_model(input_shape=(48, 48, 1), num_classes=4):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert('L').resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0
    return img_array


def launch_gui(model, history, acc):
    def generate_plot(history):
        # If history is a dict (from pickle), use it directly
        if isinstance(history, dict):
            hist = history
        # If history is a keras History object, use its .history attribute
        elif hasattr(history, 'history'):
            hist = history.history
        else:
            result_label.config(text="Invalid history format.")
            return

        plt.figure(figsize=(7, 4))
        if 'accuracy' in hist and 'val_accuracy' in hist:
            plt.plot(hist['accuracy'], label='Train Accuracy')
            plt.plot(hist['val_accuracy'], label='Val Accuracy')
        if 'loss' in hist and 'val_loss' in hist:
            plt.plot(hist['loss'], label='Train Loss')
            plt.plot(hist['val_loss'], label='Val Loss')
        plt.legend()
        plt.title("Training vs Validation Accuracy/Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    def retrain():
        nonlocal model
        train_gen, val_gen, test_gen = load_data(BASE_DIR)
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=50)

        model.save(MODEL_PATH)
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        _, new_acc = model.evaluate(test_gen)
        acc_label.config(text=f"Current Model Accuracy : {new_acc:.2f}")
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
                    if file.lower().endswith(('.jpg', '.png', '.jpeg', '.avif')):
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
    root.geometry("370x560")
    root.configure(bg="#fbffcc")

    # Style settings
    BUTTON_BG = "#4a90e2"
    BUTTON_FG = "#ffffff"
    BUTTON_ACTIVE_BG = "#357ab8"
    LABEL_BG = "#fbffcc"
    LABEL_FG = "#222222"
    RESULT_FG = "#e94e77"

    # Title label
    title_label = tk.Label(root, text="Facial Expression Classifier", font=("Arial", 18, "bold"),
                           bg=LABEL_BG, fg="#2d3e50")
    title_label.pack(pady=(18, 8))

    # Frame for buttons
    btn_frame = tk.Frame(root, bg=LABEL_BG)
    btn_frame.pack(pady=8)

    choose_btn = tk.Button(
        btn_frame, text="Choose Image", command=classify_image,
        bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_ACTIVE_BG,
        font=("Arial", 11, "bold"), width=14, relief="raised", bd=2, cursor="hand2"
    )
    choose_btn.grid(row=0, column=0, padx=7, pady=4)

    testall_btn = tk.Button(
        btn_frame, text="Test All Classes", command=test_all,
        bg=BUTTON_BG, fg=BUTTON_FG, activebackground=BUTTON_ACTIVE_BG,
        font=("Arial", 11, "bold"), width=14, relief="raised", bd=2, cursor="hand2"
    )
    testall_btn.grid(row=0, column=1, padx=7, pady=4)

    retrain_btn = tk.Button(
        root, text="Retrain Model", command=retrain,
        bg="#50c878", fg=BUTTON_FG, activebackground="#3e8e5a",
        font=("Arial", 11, "bold"), width=32, relief="raised", bd=2, cursor="hand2"
    )
    retrain_btn.pack(pady=6)

    plot_btn = tk.Button(
        root, text="Show Training Plot", command=lambda: generate_plot(history),
        bg="#f5a623", fg=BUTTON_FG, activebackground="#c97d0d",
        font=("Arial", 11, "bold"), width=32, relief="raised", bd=2, cursor="hand2"
    )
    plot_btn.pack(pady=6)

    exit_btn = tk.Button(
        root, text="Exit", command=root.quit,
        bg="#e94e77", fg=BUTTON_FG, activebackground="#b83250",
        font=("Arial", 11, "bold"), width=32, relief="raised", bd=2, cursor="hand2"
    )
    exit_btn.pack(pady=6)

    acc_label = tk.Label(root, text=f"Current Accuracy : {acc:.2f}",
                         font=("Arial", 12, "bold"), bg=LABEL_BG, fg="#2d3e50")
    acc_label.pack(pady=(10, 2))

    current_class_label = tk.Label(root, text="Current Class Testing: None",
                                   font=("Arial", 11), bg=LABEL_BG, fg=LABEL_FG)
    current_class_label.pack()

    image_label = tk.Label(root, bg=LABEL_BG)
    image_label.pack(pady=10)

    result_label = tk.Label(root, text="", font=("Arial", 15, "bold"),
                            bg=LABEL_BG, fg=RESULT_FG)
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
    else: #train for the first time
        model = build_model()
        history = model.fit(train_gen, validation_data=val_gen, epochs=40)
        model.save(MODEL_PATH)
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(history.history, f)
        _, acc = model.evaluate(test_gen)

    launch_gui(model, history, acc)


if __name__ == "__main__":
    main()
