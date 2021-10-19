import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model

data_test = "./data"
model_path = "./model.h5"

batch_size = 32
n_classes = 75
img_width, img_height = 299, 299

model = load_model(model_path)

test_datagen = image.ImageDataGenerator(rescale=1./255)

test_flow = test_datagen.flow_from_directory(
    data_test,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

n_test = test_flow.n

print(n_test)

model.compile(optimizer='nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

loss, accuracy = model.evaluate_generator(
    test_flow,
    steps=n_test//batch_size,
    verbose=1)

print("\nLoss =", loss)
print("Accuracy =", accuracy)