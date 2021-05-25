import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from collections import Counter

# dimensions of our images
img_width, img_height = 299, 299

# load the model we saved
model = load_model('./model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

pred_list = []
class_nums = []
# predicting images
for i1 in range (21,96):
    print("----> ", i1)
    directory = "./data/" + str(i1)
    for filename in os.listdir(directory):
        imgpath = os.path.join(directory, filename)

        if filename.split('.')[-1] != 'jpg':
            continue

        img = image.load_img(imgpath, target_size=(img_width, img_height))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x, axis=0)

        predictions = model.predict(x, batch_size=10)
        
        outputclass = np.argmax(predictions) + 21

        print(i1, "-", outputclass)
        #print(str(i1) + " - " + str(outputclass))
        pred_list.append(outputclass)
        class_nums.append(i1)



preds_counted = Counter(pred_list)
class_counted = Counter(class_nums)

print(preds_counted)
print(class_counted)
