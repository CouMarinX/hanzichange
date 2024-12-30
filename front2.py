from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# 初始化Flask应用
app = Flask(__name__)

# 自定义 TransformerBlock
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 加载TensorFlow模型
from tensorflow.keras.losses import MeanSquaredError
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'mse': MeanSquaredError()
}
model = tf.keras.models.load_model('transformer_model.h5', custom_objects=custom_objects)

# 将汉字转为16x16二维数组的示例函数
def char_to_16x16_array(char):
    array = np.zeros((16, 16))
    for i in range(min(len(char), 16)):
        array[i, i] = ord(char[i]) % 256
    return array

# 将二维数组转为图片
def array_to_image(array):
    array_normalized = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    image = Image.fromarray(array_normalized.astype('uint8'))
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    chars = data.get('chars', '')

    # 转换汉字为二维数组并进行预测
    images = []
    for char in chars:
        input_array = char_to_16x16_array(char)
        input_array = input_array.reshape(1, 16, 16, 1)  # 调整为模型输入维度
        predicted_array = model.predict(input_array)
        predicted_array = predicted_array.squeeze()

        # 转换输出为图片
        image = array_to_image(predicted_array)
        images.append(image)

    # 拼接图片
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    result_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # 保存图片
    output_path = os.path.join('static', 'result.png')
    result_image.save(output_path)

    return jsonify({'image_url': output_path})

if __name__ == '__main__':
    app.run(debug=True)
