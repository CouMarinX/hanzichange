import tkinter as tk
from tkinter import filedialog, colorchooser, font, ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk
import os


def text_to_image(text, font_path, image_path, font_size=32, color=(0, 0, 0)):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}")
        return None

    # 使用 getbbox 获取文本边界框
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    image = Image.new("RGB", (text_width + 20, text_height + 20), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=color)
    image.save(image_path)
    return image


def generate_image():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        return

    font_path = font_var.get()
    font_size = size_var.get()
    color = color_var.get()
    image_path = "output.png"

    image = text_to_image(text, font_path, image_path, font_size, color)
    if image:
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo # 保持引用，防止垃圾回收

def choose_font():
    font_path = filedialog.askopenfilename(filetypes=[("Font Files", "*.ttf;*.otf")])
    if font_path:
        font_var.set(font_path)
        font_label.config(text=f"当前字体：{os.path.basename(font_path)}")

def choose_color():
    color = colorchooser.askcolor(title="选择颜色")
    if color[1]:
        color_var.set(color[1])
        color_preview.config(bg=color[1])

root = tk.Tk()
root.title("文字转图片工具")

# 使用 PanedWindow 实现可调整大小的分割
paned_window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# 输入框架
input_frame = ttk.Frame(paned_window, padding=10)
paned_window.add(input_frame, weight=1) # weight 决定了框架的相对大小

# 文本输入
text_label = tk.Label(input_frame, text="输入文本：")
text_label.grid(row=0, column=0, sticky="w") # sticky="w" 左对齐
text_entry = tk.Text(input_frame, height=5)
text_entry.grid(row=1, column=0, sticky="nsew") # sticky="nsew" 填充整个单元格

# 字体选择
font_var = tk.StringVar()
font_label = tk.Label(input_frame, text="未选择字体")
font_label.grid(row=2, column=0, sticky="w")
font_button = tk.Button(input_frame, text="选择字体", command=choose_font)
font_button.grid(row=3, column=0, sticky="ew")

# 字体大小
size_label = tk.Label(input_frame, text="字体大小：")
size_label.grid(row=4, column=0, sticky="w")
size_var = tk.IntVar(value=32)
size_entry = tk.Entry(input_frame, textvariable=size_var)
size_entry.grid(row=5, column=0, sticky="ew")

# 颜色选择
color_var = tk.StringVar(value="#000000")
color_label = tk.Label(input_frame, text="字体颜色：")
color_label.grid(row=6, column=0, sticky="w")
color_preview = tk.Label(input_frame, bg=color_var.get(), width=10)
color_preview.grid(row=7, column=0, sticky="ew")
color_button = tk.Button(input_frame, text="选择颜色", command=choose_color)
color_button.grid(row=8, column=0, sticky="ew")


# 按钮框架
button_frame = ttk.Frame(paned_window, padding=10)
paned_window.add(button_frame, weight=0)

# 生成图片按钮
generate_button = tk.Button(button_frame, text="生成图片", command=generate_image)
generate_button.pack(pady=20) # 添加垂直方向的间距

# 图片显示框架
image_frame = ttk.Frame(paned_window, padding=10)
paned_window.add(image_frame, weight=1)

# 图片显示区域
image_label = tk.Label(image_frame)
image_label.pack(fill=tk.BOTH, expand=True) # 填充框架

# 配置行和列的权重，使输入框和图片区域可以拉伸
input_frame.columnconfigure(0, weight=1)
input_frame.rowconfigure(1, weight=1)
image_frame.columnconfigure(0, weight=1)
image_frame.rowconfigure(0, weight=1)

root.mainloop()
