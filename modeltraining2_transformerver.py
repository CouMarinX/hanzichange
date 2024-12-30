import tensorflow as tf  # 导入TensorFlow库
from tensorflow.keras.models import Model  # type: ignore # 导入Keras模型基类
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout  # type: ignore # 导入必要的层
from tensorflow.keras.optimizers import Adam  # type: ignore # 导入Adam优化器
import numpy as np  # 导入NumPy库
import ast  # 导入ast库解析txt格式

# Read dataset from files
def load_data(input_file, target_file):  # 从文件加载数据
    with open(input_file, "r") as f:  # 打开输入文件
        inputs = np.array(ast.literal_eval(f.read()), dtype=np.float32)  # 使用ast解析并转换为NumPy数组
    with open(target_file, "r") as f:  # 打开目标文件
        targets = np.array(ast.literal_eval(f.read()), dtype=np.float32)  # 使用ast解析并转换为NumPy数组
    return inputs, targets

# Custom Transformer block
class TransformerBlock(tf.keras.layers.Layer):  # 定义Transformer模块
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # 初始化函数
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # 多头注意力层
        self.ffn = tf.keras.Sequential([  # 前向传播网络
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # 第一层归一化
        self.layernorm2 = LayerNormalization(epsilon=1e-6)  # 第二层归一化
        self.dropout1 = Dropout(rate)  # 第一层Dropout
        self.dropout2 = Dropout(rate)  # 第二层Dropout

    def call(self, inputs, training=False):  # 前向传播函数，默认training为False
        attn_output = self.att(inputs, inputs)  # 计算注意力
        attn_output = self.dropout1(attn_output, training=training)  # 应用Dropout
        out1 = self.layernorm1(inputs + attn_output)  # 残差连接和归一化
        ffn_output = self.ffn(out1)  # 前向传播网络输出
        ffn_output = self.dropout2(ffn_output, training=training)  # 应用Dropout
        return self.layernorm2(out1 + ffn_output)  # 残差连接和归一化

# Transformer-based model
def build_model(input_shape, output_dim, num_heads=4, num_layers=2, ff_dim=64, rate=0.1):  # 构建模型函数
    inputs = Input(shape=input_shape)  # 定义输入层
    x = inputs  # 初始化中间层
    for _ in range(num_layers):  # 添加多个Transformer模块
        x = TransformerBlock(embed_dim=input_shape[-1], num_heads=num_heads, ff_dim=ff_dim, rate=rate)(x, training=True)
    outputs = Dense(output_dim)(x)  # 输出层
    model = Model(inputs=inputs, outputs=outputs)  # 定义模型
    return model

# Training function
def train_model(model, inputs, targets, epochs=10, batch_size=1):  # 定义训练模型的函数
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")  # 编译模型，使用均方误差损失函数
    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, verbose=1)  # 训练模型

# Main entry
if __name__ == "__main__":  # 主函数入口
    # Load data from files
    input_file = "output/base_character.txt"  # 输入文件路径
    target_file = "output/new_character.txt"  # 目标文件路径
    inputs, targets = load_data(input_file, target_file)  # 加载数据

    # Model
    model = build_model(input_shape=(16, 16), output_dim=16)  # 构建Transformer模型，设置输入输出维度

    # Train the model
    train_model(model, inputs, targets, epochs=10, batch_size=16)  # 调用训练函数，开始训练模型

    # Save the trained model
    model.save("transformer_model.h5")  # 保存模型到当前文件夹，文件名为transformer_model.h5

    print("Model saved as transformer_model.h5")
