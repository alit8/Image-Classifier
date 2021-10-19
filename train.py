# Autur: Ali Saberi
# Email: ali.saberi96@gmail.com


########## A General Image Classifier ##########

#-----> Creating train and validation split ---> comment these 3 lines if you have your own train and validation data
# import splitfolders
# data_path = "./data" # path to directory containing class folders
# splitfolders.ratio(data_path, output="data_split", seed=1337, ratio=(.8, .2), group_prefix=None)

import os
from tensorflow.keras.applications import Xception, ResNet50, NASNetLarge, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD, Nadam
from matplotlib import pyplot as plt

data_train = "./data_split/train"
data_val = "./data_split/val"
model_dir_path = "." # path to directory where model weights are saved
top_weights_path = os.path.join(os.path.abspath(model_dir_path), 'top_model_weights.h5')
final_weights_path = os.path.join(os.path.abspath(model_dir_path), 'model_weights.h5')
model_path = os.path.join(os.path.abspath(model_dir_path), 'model.h5')

n_classes = 75 # number of classes
# based_model_last_block_layer_number = 403 # for InceptionResNetV2
based_model_last_block_layer_number = 36
# img_width, img_height = 331, 331
img_width, img_height = 299, 299
# img_width, img_height = 224, 224
batch_size = 32
n_epoch = 50
lr = 1e-5
momentum = 0.9

#-----> Creating model
base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
# x = Dense(2048, activation='relu')(x)
# x = Dropout(0.5)(x)
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for i, layer in enumerate(base_model.layers):
    layer.trainable = False
    print(i, layer.name)

# print(model.summary())

# base_model = InceptionResNetV2(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(2048, activation='relu')(x)
# x = Dropout(0.5)(x)
# # x = Dense(2048, activation='relu')(x)
# # x = Dropout(0.5)(x)
# predictions = Dense(n_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# for i, layer in enumerate(base_model.layers):
#     layer.trainable = False
#     print(i, layer.name)

#print(model.summary())

#------> Preparing data
train_datagen = image.ImageDataGenerator(rescale=1./255,
                             rotation_range=5,
                             shear_range=0.1,
                             zoom_range=0.1,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             brightness_range=[0.7,1.3],
                             horizontal_flip=True)

val_datagen = image.ImageDataGenerator(rescale=1./255)

train_flow = train_datagen.flow_from_directory(
    data_train,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_flow = val_datagen.flow_from_directory(
    data_val,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

n_train = train_flow.n
n_val = val_flow.n


#------> Training classifier
model.compile(optimizer='nadam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(top_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_accuracy', patience=5, verbose=0)
]

history = model.fit_generator(train_flow,
                    steps_per_epoch=n_train//batch_size,
                    epochs=int(n_epoch),
                    validation_data=val_flow,
                    validation_steps=n_val//batch_size,
                    callbacks=callbacks_list)

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

#------> Fine-tuning model
model.load_weights(top_weights_path)

for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

# opt = SGD(learning_rate=lr, momentum=momentum)
opt = Nadam(learning_rate=lr)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

history = model.fit_generator(train_flow,
                    steps_per_epoch=n_train//batch_size,
                    epochs=n_epoch,
                    validation_data=val_flow,
                    validation_steps=n_val//batch_size,
                    callbacks=callbacks_list)

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#-----> Saving model
model_json = model.to_json()
with open(os.path.join(os.path.abspath(model_dir_path), 'model.json'), 'w') as json_file:
    json_file.write(model_json)

model.save(model_path)