
# import required packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization


def plot_curves(history):
    print("plotting curves")
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    plt.figure(figsize=(15,5))

    #plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label = "training_loss")
    plt.plot(epochs, val_loss, label = "val_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label = "training_accuracy")
    plt.plot(epochs, val_accuracy, label = "val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()
    
def Create_CNN_Model():
    
    model = Sequential()
    
    #CNN1
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    
    #CNN2
    model.add(Conv2D(64, (3,3), activation='relu', ))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    
    #CNN3
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.25))
    
    
    #Output
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(7,activation='softmax'))
    
    
    return model

def fer_improved_cnn():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))

    return model

def fer_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    return model

def fer_model_256():
        model = Sequential()
        #1st convo
        model.add(Conv2D(96, (3,3),activation='relu' ,input_shape = (48,48,1)))
        #polling
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        #2nd convo
        model.add(Conv2D(256, (3,3),activation='relu'))
        #polling
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        #3rd convo
        model.add(Conv2D(384, (3,3),activation='relu'))
        #polling
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        #4th convo
        model.add(Conv2D(256, (3,3),activation='relu'))
        #polling
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        #passing through dense layer
        model.add(Flatten())

        #1st dense layer
        model.add(Dense(1024,activation='relu'))
        #dropout
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        #2nd dense layer
        model.add(Dense(5,activation='softmax'))
        model.add(Activation('relu'))
        #dropout
        model.add(BatchNormalization(),Dropout(0.4))

        #3rd dense layer
        model.add(Dense(256,activation='relu'))
        #dropout
        model.add(BatchNormalization(),Dropout(0.4),Activation('relu'))

        #output layer
        model.add(Dense(5),Activation('softmax'))
        return model
# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,                                        
        fill_mode='nearest')
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data-clean/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        shuffle='false',
        class_mode='categorical')


# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data-clean/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        shuffle='false',
        class_mode='categorical')

cv2.ocl.setUseOpenCL(False)


model = fer_model()

# early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')

# model check point
mc = ModelCheckpoint(filepath="results/model/latestModel-7.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')

# Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning rate
rlr = ReduceLROnPlateau( monitor='val_loss',
                                                  factor=0.2,
                                                  patience=2,
#                                                 min_lr=0.000005,
                                                  verbose=1)

# puting call back in a list
call_back = [es, mc]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the neural network/model
model_info = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=50,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=call_back)

plot_curves(model_info)

CNN_Predictions = model.predict(validation_generator)

# Choosing highest probalbilty class in every prediction 
CNN_Predictions = np.argmax(CNN_Predictions, axis=1)

import seaborn as sns 
from sklearn.metrics import confusion_matrix

fig, ax= plt.subplots(figsize=(15,10))

cm=confusion_matrix(validation_generator.labels, CNN_Predictions)

ax.set_xlabel('Predicted labels',fontsize=15, fontweight='bold')
ax.set_ylabel('True labels', fontsize=15, fontweight='bold')
ax.set_title('CNN Confusion Matrix', fontsize=20, fontweight='bold')
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
plt.show()
