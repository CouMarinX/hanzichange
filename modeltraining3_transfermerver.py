import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import ast

# Read dataset from files
def load_data(input_file, target_file):
    with open(input_file, "r") as f:
        inputs = np.array(ast.literal_eval(f.read()), dtype=np.float32)
    with open(target_file, "r") as f:
        targets = np.array(ast.literal_eval(f.read()), dtype=np.float32)
    return inputs, targets

# Custom Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer-based model
def build_model(input_shape, output_dim, num_heads=4, num_layers=2, ff_dim=64, rate=0.1):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim=input_shape[-1], num_heads=num_heads, ff_dim=ff_dim, rate=rate)(x, training=True)
    outputs = Dense(output_dim)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Training function
def train_model(model, inputs, targets, epochs=10, batch_size=1):
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, verbose=1)

# Main entry
if __name__ == "__main__":
    input_file = "output/base_character.txt"
    target_file = "output/new_character.txt"
    inputs, targets = load_data(input_file, target_file)

    model = build_model(input_shape=(16, 16), output_dim=16)
    train_model(model, inputs, targets, epochs=10, batch_size=16)
    model.save("transformer_model1.keras")
    print("Model saved as transformer_model1.keras")