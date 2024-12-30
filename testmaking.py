import os
from hanzi_chaizi import HanziChaizi
from PIL import Image, ImageDraw, ImageFont

# 初始化汉字拆字工具
chaizi_tool = HanziChaizi()

def decompose_hanzi(character):
    """拆解汉字为部件"""
    try:
        components = chaizi_tool.query(character)
        if components:
            return components[0]  # 返回拆分后的第一个结果
    except Exception as e:
        print(f"Error decomposing {character}: {e}")
    return []

def generate_bitmap(character, font_path='simsun.ttc', size=(16, 16)):
    """生成汉字的16x16位图表示"""
    try:
        font = ImageFont.truetype(font_path, 16)
        image = Image.new('1', size, 1)  # 创建白底黑字的图片
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, font=font, fill=0)
        return [[1 if image.getpixel((x, y)) == 0 else 0 for x in range(size[0])] for y in range(size[1])]
    except Exception as e:
        print(f"Error generating bitmap for {character}: {e}")
        return []

def find_related_hanzi(base_character, characters):
    """查找由base_character加偏旁部首后形成的汉字"""
    base_components = decompose_hanzi(base_character)

    if not base_components:
        # 如果原汉字无法拆解，直接找包含它的字符
        for char in characters:
            char_components = decompose_hanzi(char)
            if base_character in char_components:
                return char  # 找到第一个包含的字符即返回
    else:
        for char in characters:
            char_components = decompose_hanzi(char)
            if base_character in char_components and base_components != char_components:
                return char  # 找到第一个匹配的字符即返回

    return None

def save_results_as_array(results, output_dir="output"):
    """保存结果为二维数组的txt文件"""
    os.makedirs(output_dir, exist_ok=True)

    base_file = os.path.join(output_dir, "base_character.txt")
    new_file = os.path.join(output_dir, "new_character.txt")

    with open(base_file, "w", encoding="utf-8") as bf:
        bf.write("[\n")
        for base, _ in results:
            bitmap = generate_bitmap(base)
            bf.write("    [\n")
            for row in bitmap:
                bf.write("        [" + ", ".join(map(str, row)) + "],\n")
            bf.write("    ],\n")
        bf.write("]\n")

    with open(new_file, "w", encoding="utf-8") as nf:
        nf.write("[\n")
        for _, new_char in results:
            if new_char:
                bitmap = generate_bitmap(new_char)
                nf.write("    [\n")
                for row in bitmap:
                    nf.write("        [" + ", ".join(map(str, row)) + "],\n")
                nf.write("    ],\n")
        nf.write("]\n")

if __name__ == "__main__":
    characters_to_check = [chr(i) for i in range(0x4e00, 0x9e00)]  # 缩小常用汉字范围进行测试
    results = []

    print("Processing, this might take a while...")
    for base_character in characters_to_check:
        related_character = find_related_hanzi(base_character, characters_to_check)
        if related_character:
            results.append((base_character, related_character))

    print(f"Processed {len(results)} base characters with related characters.")
    save_results_as_array(results)
    print("Results saved in the output directory.")
