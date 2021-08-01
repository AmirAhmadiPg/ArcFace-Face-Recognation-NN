from tensorflow.keras import layers

def OneAttentionBlock(PLayer, n_filters):

    key = layers.Conv2D(n_filters, (1, 1), activation=None, use_bias=False)(PLayer)

    query = layers.Conv2D(n_filters, (1, 1), activation=None, use_bias=False)(PLayer)

    values = layers.Conv2D(1, (1, 1), activation=None, use_bias=False)(PLayer)

    attention_kernel = layers.Multiply()([key, query])

    attention_kernel_normilized = layers.Softmax()(attention_kernel)

    attention_output = layers.Multiply()([attention_kernel_normilized, values])

    return attention_output


