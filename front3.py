
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import tensorflow as tf
from modeltraining2_transformerver import TransformerBlock
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout

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
    
# 加载模型
try:
    model = tf.keras.models.load_model('transformer_model.keras', custom_objects={'TransformerBlock': TransformerBlock})
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

# 汉字转16x16二维数组
def hanzi_to_array(hanzi):
    font = ImageFont.truetype('仿宋_gb2312.ttf', 16)  # 请替换为实际的字体文件路径
    image = Image.new('L', (16, 16), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), hanzi, font=font, fill=0)
    array = np.array(image)
    array = (array < 128).astype(int)
    return array

# 处理输入汉字
def process_hanzi():
    text = text_input.get('1.0', tk.END).strip()
    if not text:
        messagebox.showwarning("警告", "请输入汉字！")
        return
    if model is None:
        messagebox.showwarning("警告", "模型加载失败！")
        return

    hanzi_list = list(text)
    input_arrays = [hanzi_to_array(hanzi) for hanzi in hanzi_list]
    input_data = np.array(input_arrays)
    input_data = input_data.reshape(-1, 16, 16, 1)

    # 使用模型进行预测
    output_data = model.predict(input_data)
    output_data = output_data.reshape(-1, 16, 16)

    # 将输出数组转为图片并拼接
    images = []
    for arr in output_data:
        img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(img)

    # 拼接图片
    width = 16 * len(images)
    height = 16
    total_image = Image.new('L', (width, height))
    for i, img in enumerate(images):
        total_image.paste(img, (i*16, 0))

    # 显示图片
    photo = ImageTk.PhotoImage(total_image)
    image_label.config(image=photo)
    image_label.image = photo

    # 保存图片
    def save_image():
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg")])
        if file_path:
            total_image.save(file_path)

    # 更新保存按钮的命令
    save_button.config(command=save_image)

# 创建主窗口
root = tk.Tk()
root.title("汉字转换器")

# 左侧栏
left_frame = tk.Frame(root)
left_frame.pack(side='left', fill='both', expand=True)

# 输入汉字
text_label = tk.Label(left_frame, text="输入汉字:")
text_label.pack()
text_input = tk.Text(left_frame, height=10, width=30)
text_input.pack()

# 中间栏
middle_frame = tk.Frame(root)
middle_frame.pack(side='left', fill='both', expand=True)

# 右侧栏
right_frame = tk.Frame(root)
right_frame.pack(side='right', fill='both', expand=True)

# 显示图片
image_label = tk.Label(right_frame)
image_label.pack(expand=True, fill='both')

# 保存按钮
save_button = tk.Button(right_frame, text="保存图片", state='disabled')
save_button.pack(pady=10)

# 处理按钮
process_button = tk.Button(left_frame, text="生成图片", command=process_hanzi)
process_button.pack(pady=10)

# 运行主循环
root.mainloop()