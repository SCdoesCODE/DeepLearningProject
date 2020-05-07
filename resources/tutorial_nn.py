"""
From the sentdex tutorial
https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist #28*28 sized images of handwritten digits 0-9

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#normalize the data : scale on 0-1, easier to learn
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#sequential model : do things in direct order, feedforward
model = tf.keras.models.Sequential()
#add the input layer
model.add(tf.keras.layers.Flatten())
#add two hidden layers with 128 nodes and the RELU activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#output layer and the activation function to be softmax, because we want a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#we pass in the optimizer method, loss method, and also which metrics we will track
#you could pass in, instead of adam, e.g. sgd, but adam is kinda the default goto, like relu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs=3)

#to make sure that we didn't overfit, also measure validation loss and accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

#save the model
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')

#make predictions on the test set
#will return probability distributions, not very readable
predictions = new_model.predict(x_test)

#this should print the actual prediction
print(np.argmax(predictions[0]))

#this image should depict the same number as the prediction
plt.imshow(x_train[0], cmap = plt.cm.binary) #cmap makes it a non-color image
plt.show()



