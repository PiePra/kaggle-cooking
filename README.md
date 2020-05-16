# kaggle-cooking
kaggles What's Cooking? challenge with keras/tensorflow 

training accuracy: 86%
test accuracy: 67%

```python
import json
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import pandas as pd
import numpy as np
from random import randrange

#data subdirectory
dir = './data/'

#open input training file
with open(dir + 'train.json','r') as f:
    train_file = json.load(f)

#open input test file
with open(dir + 'test.json','r') as f:
    test_file = json.load(f)

```

    Using TensorFlow backend.
    


```python
##lists for every input
#cuisines of train file
cuisines_train = [line['cuisine'] for line in train_file]
cuisines = list(set(cuisines_train))
#ingredients list of list of train file
ingredients_train = [line['ingredients'] for line in train_file]
#ids of train file
ids_train = [line['id'] for line in train_file]
#ids of test file
ids_test = [line['id'] for line in test_file]
#ingredients list of list of test file
ingredients_test = [line['ingredients'] for line in test_file]
#all ingredients in one list of list
ingredients_all = ingredients_train + ingredients_test
ingredients_flat = list(set([element for line in ingredients_all for element in line]))

```


```python
cuisines_vector=[]
#get cuisines input vector
for line in cuisines_train:
    temp = np.zeros(len(cuisines))
    temp[cuisines.index(line)] = 1
    cuisines_vector.append(temp)
    
    
```


```python
#get ingredients input vector
ingredients_vector=[]
for line in ingredients_train:
    temp = np.zeros(len(ingredients_flat))
    for item in line:
        temp[ingredients_flat.index(item)] = 1
    ingredients_vector.append(temp)
```


```python

#create keras model
model = Sequential()
#7137 input neurons ( count of unique ingredients)
#print(len(ingredient_map))
model.add(Dense(50, activation="relu", input_dim=7137))
model.add(Dense(25, activation="relu"))
model.add(Dense(20, activation="relu"))
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
#train model, 50 epos, 
model.fit(np.array(ingredients_vector),np.array(cuisines_vector),epochs=35,batch_size=8)

```

    Epoch 1/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0237 - accuracy: 0.6569
    Epoch 2/35
    39774/39774 [==============================] - 43s 1ms/step - loss: 0.0184 - accuracy: 0.7276
    Epoch 3/35
    39774/39774 [==============================] - 46s 1ms/step - loss: 0.0163 - accuracy: 0.7542
    Epoch 4/35
    39774/39774 [==============================] - 49s 1ms/step - loss: 0.0148 - accuracy: 0.7721
    Epoch 5/35
    39774/39774 [==============================] - 44s 1ms/step - loss: 0.0138 - accuracy: 0.7836
    Epoch 6/35
    39774/39774 [==============================] - 48s 1ms/step - loss: 0.0132 - accuracy: 0.7900
    Epoch 7/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0127 - accuracy: 0.7957
    Epoch 8/35
    39774/39774 [==============================] - 43s 1ms/step - loss: 0.0124 - accuracy: 0.7994
    Epoch 9/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0121 - accuracy: 0.8022
    Epoch 10/35
    39774/39774 [==============================] - 45s 1ms/step - loss: 0.0119 - accuracy: 0.8041
    Epoch 11/35
    39774/39774 [==============================] - 45s 1ms/step - loss: 0.0117 - accuracy: 0.8057
    Epoch 12/35
    39774/39774 [==============================] - 47s 1ms/step - loss: 0.0116 - accuracy: 0.8066
    Epoch 13/35
    39774/39774 [==============================] - 43s 1ms/step - loss: 0.0114 - accuracy: 0.8078
    Epoch 14/35
    39774/39774 [==============================] - 41s 1ms/step - loss: 0.0113 - accuracy: 0.8089
    Epoch 15/35
    39774/39774 [==============================] - 44s 1ms/step - loss: 0.0113 - accuracy: 0.8092
    Epoch 16/35
    39774/39774 [==============================] - 44s 1ms/step - loss: 0.0112 - accuracy: 0.8099
    Epoch 17/35
    39774/39774 [==============================] - 44s 1ms/step - loss: 0.0111 - accuracy: 0.8108
    Epoch 18/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0110 - accuracy: 0.8113
    Epoch 19/35
    39774/39774 [==============================] - 48s 1ms/step - loss: 0.0110 - accuracy: 0.8116
    Epoch 20/35
    39774/39774 [==============================] - 48s 1ms/step - loss: 0.0110 - accuracy: 0.8123
    Epoch 21/35
    39774/39774 [==============================] - 41s 1ms/step - loss: 0.0109 - accuracy: 0.8132
    Epoch 22/35
    39774/39774 [==============================] - 40s 1ms/step - loss: 0.0108 - accuracy: 0.8139
    Epoch 23/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0108 - accuracy: 0.8136
    Epoch 24/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0108 - accuracy: 0.8137
    Epoch 25/35
    39774/39774 [==============================] - 40s 996us/step - loss: 0.0108 - accuracy: 0.8144
    Epoch 26/35
    39774/39774 [==============================] - 40s 1ms/step - loss: 0.0107 - accuracy: 0.8148
    Epoch 27/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0107 - accuracy: 0.8148
    Epoch 28/35
    39774/39774 [==============================] - 46s 1ms/step - loss: 0.0107 - accuracy: 0.8148
    Epoch 29/35
    39774/39774 [==============================] - 41s 1ms/step - loss: 0.0107 - accuracy: 0.8155
    Epoch 30/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0106 - accuracy: 0.8154
    Epoch 31/35
    39774/39774 [==============================] - 45s 1ms/step - loss: 0.0106 - accuracy: 0.8153
    Epoch 32/35
    39774/39774 [==============================] - 45s 1ms/step - loss: 0.0106 - accuracy: 0.8158
    Epoch 33/35
    39774/39774 [==============================] - 42s 1ms/step - loss: 0.0106 - accuracy: 0.8157
    Epoch 34/35
    39774/39774 [==============================] - 46s 1ms/step - loss: 0.0106 - accuracy: 0.8157
    Epoch 35/35
    39774/39774 [==============================] - 44s 1ms/step - loss: 0.0105 - accuracy: 0.8161
    




    <keras.callbacks.callbacks.History at 0x258cea436c8>




```python
ingredients_test_vector=[]
for line in ingredients_test:
    temp = np.zeros(len(ingredients_flat))
    for item in line:
        temp[ingredients_flat.index(item)] = 1
    ingredients_test_vector.append(temp)
```


```python
preds = model.predict(np.array(ingredients_test_vector))
```


```python
submit = []
for line in preds:
    result = np.where(line == np.amax(line))
    if len(result[0]) == 1:
        submit.append(cuisines[int(result[0])])
    else: 
        submit.append(cuisines[randrange(20)])
```


```python
len(submit)
```




    9944




```python
with open ('submit.csv','w') as w:
    for i in range (0,len(submit)):
        w.write(str(ids_test[i]) + ',' + str(submit[i]) + '\n' )
    
```
