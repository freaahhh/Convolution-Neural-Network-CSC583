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

#LOAD DATA INTO MODEL

def load_data(base_dir, selected_classes):

    # Data augmentation technique for training
    train_datagen = ImageDataGenerator(
        rescale=1./255, #Normalizaiton chg color
        horizontal_flip=True, #image flip
        rotation_range=10, #rotate
        zoom_range=0.1, #zoom

        validation_split=0.2  # reserve part of training for validation

        # # other technique that lower the accuracy
        # width_shift_range=0.1
        # height_shift_range=0.1
        # shear_range=0.1
        # brightness_range=(0.8, 1.2)
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(48, 48),
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=25,
        shuffle=True,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(48, 48),
        color_mode='grayscale',
        classes=selected_classes,
        class_mode='categorical',
        batch_size=25,
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
        batch_size=25,
        shuffle=False
    )

    # testing
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")

    return train_generator, val_generator, test_generator

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)), #Filter 1 
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'), #Filter 2
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'), #Filter 3
        MaxPooling2D(2,2),

        Conv2D(256, (3,3), activation='relu'), #Filter 3
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary() testing
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

    model.fit(train_gen, validation_data=val_gen, epochs=150)

    loss, acc = model.evaluate(test_gen) #test the 5 image
    print(f"Test accuracy: {acc:.2f}")
    

if __name__ == "__main__":
    main()
