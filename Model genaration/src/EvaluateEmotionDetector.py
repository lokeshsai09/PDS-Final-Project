
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay



emotion_dict = {0: "Angry", 1: "Fear", 2: "Happy", 3: "Neutral", 4: "Sad"}

#emotion_dict = {0:"Angry",1:"Disguist",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}
# load weights into new model
model = load_model("results/model/latestModel.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        'data-clean/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        #color_mode='rgb',
        shuffle='false',
        class_mode='categorical')

# do prediction on test data
predictions = model.predict(test_generator)

predictions = np.argmax(predictions, axis=1)

print("-----------------------------------------------------------------")
# confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions,normalize='true')
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)

cm_display.plot(cmap=plt.cm.winter)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))




