from arcface_layer import ArcFace
from tensorflow.keras import regularizers
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from attention_block import Network_block
from tensorflow.keras.utils import to_categorical
import numpy as np


(X, y), _ = mnist.load_data()

y = to_categorical(y, 10)
X = np.array(X)/255
y = np.array(y)

input_layer = layers.Input(shape=(28, 28, 1))
y_input_layer = layers.Input(shape=(10,))

attention1 = Network_block(input_layer, 1)
attention2 = Network_block(attention1, 1)

flatten = layers.Flatten()(attention2)

hidden0 = layers.Dense(500, activation= 'relu')(flatten)
hidden1 = layers.Dense(150, activation= 'relu')(hidden0)

weight_decay = 1e-4

cf = ArcFace(10, regularizer=regularizers.l2(weight_decay))([hidden1, y_input_layer])

classifier = models.Model([input_layer, y_input_layer], cf)

classifier.compile(optimizer='adam',
                       loss = 'categorical_crossentropy',
                       metrics=['accuracy'])

history = classifier.fit([X, y], y, epochs = 10, batch_size=16, validation_split=0.1)