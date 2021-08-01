from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow import acos, cos, nn
from tensorflow.keras import regularizers

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
    def call(self, inputs):
        x, y = inputs

        x = nn.l2_normalize(x, axis=1)
        W = nn.l2_normalize(self.W, axis=0)

        logit = x @ W

        theta = acos(K.clip(logit, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

        target_logit = cos(theta + self.m)

        logits = logit * (1 - y) + target_logit * y

        logits *= self.s

        output = nn.softmax(logits)
        
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)