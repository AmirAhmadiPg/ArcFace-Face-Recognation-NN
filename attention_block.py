from tensorflow.keras import layers

def Network_block(PLayer, n_filters):

    Conv0 = layers.Conv2D(n_filters, (3, 3), activation=None, use_bias=False)(PLayer)

    Conv1 = layers.Conv2D(n_filters, (3, 3), activation=None, use_bias=False)(Conv0)

    Conv2 = layers.Conv2D(n_filters, (3, 3), activation=None, use_bias=False)(Conv1)


    return Conv2


