import os
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Python code implementing the 
# image recognition/classification program, including 
# data preprocessing, model building, training, evaluation and GUI.<---------------------------------

base_dir = r'C:\Users\Ahmad\Python Workspace\CSC583\training'  

# CLASS = {ANGRY,HAPPY,SAD,NEUTRAL}
selected_classes = ['angry', 'happy', 'sad', 'neutral']

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    validation_split=0.2  # reserve part of training for validation
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(48, 48),
    color_mode='grayscale',
    classes=selected_classes,
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'training'),
    target_size=(48, 48),
    color_mode='grayscale',
    classes=selected_classes,
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    subset='validation'
)

# Test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(48, 48),
    color_mode='grayscale',
    classes=selected_classes,
    class_mode='categorical',
    batch_size=64,
    shuffle=False
)

# MODEL CONVOLUTIONAL NEURAL NETWORK
from tensorflow import keras
from keras import Sequential

from tensorflow import keras
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#LOAD DATA INTO MODEL

def load_data(base_dir, selected_classes):
    # Data augmentation & preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(48, 48),
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(48, 48),
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        subset='validation'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(48, 48),
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=64,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # HAPPY,SAD,ANGRY,NEUTRAL
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# 
# 
# 
# 
# 
# MAIN HERE IMPLEMENT GUI HERE 
#
# 
# 
#        
# 

def main():
    base_dir = r'C:\Users\Ahmad\Python Workspace\CSC583\training'  #TEMP
    selected_classes = ['angry', 'happy', 'sad', 'neutral']

    train_gen, val_gen, test_gen = load_data(base_dir, selected_classes)

    model = build_model()
    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=100)

    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
