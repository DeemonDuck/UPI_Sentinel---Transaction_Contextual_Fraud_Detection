import tensorflow as tf
from tensorflow.keras import backend as K

class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        feature_dim = input_shape[-1]

        self.W = self.add_weight(
            name="W",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="b",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True
        )

        self.u = self.add_weight(
            name="u",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):

        uit = K.tanh(K.dot(inputs, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, axis=-1)

        a = K.softmax(ait)

        a_expanded = K.expand_dims(a, axis=-1)

        weighted_inputs = inputs * a_expanded

        output = K.sum(weighted_inputs, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])