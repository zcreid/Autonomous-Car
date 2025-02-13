print('Setting UP')
import os
os.environ['T_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utlis import *

#Step 1 - Initialize data
path = 'DataCollecetd'
data = importDataInfo(path)
#print(data['Center'][0])\

#Step 2 - Visualze And Balance Data
data = balanceData(data, display = True)

#Step 3 - Prepare for Proccessing
imagesPath, steerings = loadData(path, data)
# print('No of Path Creted for Image', len(imagesPath), len(steering))
# cv2.imshow('Test Image', cv2.imread(imagesPath[5]))
# cv2.waitkey(0)

#Step 4 - Split for Traning and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size =0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

# Step 5 -  Augment Data

# Step 6 - Pre Process

# Step 7 - Create Model
model = createModel()

# Step 8 - Taining
history = model.fit(dataGen(xTrain, yTrain, 100, 1), 
                    steps_per_epoch = 100,
                    epochs = 10,
                    validation_data = dataGen(xVal, yVal, 50, 0),
                    validation_steps = 50)


# Step 9 - Save The Model
model.save('model.h5')
print('Model Saved')

# Step 10 - Plot Results
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
                  