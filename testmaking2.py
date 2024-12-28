import os
from hanzi_chaizi import HanziChaizi
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
import numpy as np

# Initialize hanzi decomposition tool
chaizi_tool = HanziChaizi()

@lru_cache(maxsize=None)
def decompose_hanzi(character):
    """Decompose character into components, cached"""
    try:
        components = chaizi_tool.query(character)
        if components:
            return components[0]  # Return the first decomposition result
    except Exception as e:
        print(f"Error decomposing {character}: {e}")
    return []

def generate_bitmap(character, font, size=(16, 16)):
    """Generate 16x16 bitmap for a character using a preloaded font"""
    try:
        image = Image.new('1', size, 1)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, font=font, fill=0)
        bitmap = np.array(image)
        return (bitmap == 0).astype(int).tolist()
    except Exception as e:
        print(f"Error generating bitmap for {character}: {e}")
        return [[0]*size[0] for _ in range(size[1])]

def precompute_component_to_characters(characters, decompose_hanzi):
    """Build a mapping from components to list of characters"""
    component_to_chars = {}
    for char in characters:
        components = decompose_hanzi(char)
        for comp in components:
            if comp in component_to_chars:
                component_to_chars[comp].append(char)
            else:
                component_to_chars[comp] = [char]
    return component_to_chars

def find_related_hanzi(base_character, component_to_chars, decompose_hanzi):
    """Find a related character by base character's components"""
    base_components = decompose_hanzi(base_character)
    if not base_components:
        return None
    for comp in base_components:
        if comp in component_to_chars:
            for char in component_to_chars[comp]:
                if char != base_character and comp != char:
                    return char
    return None

def save_results_as_array(results, output_dir="output"):
    """Save base and new character bitmaps to txt files"""
    os.makedirs(output_dir, exist_ok=True)

    base_bitmaps = []
    new_bitmaps = []

    for base, new_char in results:
        base_bitmap = generate_bitmap(base, font)
        base_bitmaps.append(base_bitmap)
        if new_char:
            new_bitmap = generate_bitmap(new_char, font)
            new_bitmaps.append(new_bitmap)

    base_file = os.path.join(output_dir, "base_character.txt")
    new_file = os.path.join(output_dir, "new_character.txt")

    with open(base_file, "w", encoding="utf-8") as bf:
        bf.write("[\n")
        for bitmap in base_bitmaps:
            bf.write("    [\n")
            for row in bitmap:
                bf.write("        [" + ", ".join(map(str, row)) + "],\n")
            bf.write("    ],\n")
        bf.write("]\n")

    with open(new_file, "w", encoding="utf-8") as nf:
        nf.write("[\n")
        for bitmap in new_bitmaps:
            nf.write("    [\n")
            for row in bitmap:
                nf.write("        [" + ", ".join(map(str, row)) + "],\n")
            nf.write("    ],\n")
        nf.write("]\n")

if __name__ == "__main__":
    # Define characters to check
    characters_to_check = [chr(i) for i in range(0x4e00, 0x4e20)]  # Example range

    # Preload font
    font_path = 'simsun.ttc'
    font_size = 16
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Precompute component to characters mapping
    component_to_chars = precompute_component_to_characters(characters_to_check, decompose_hanzi)

    results = []
    invalid_cases = []

    print("Processing, this might take a while...")
    for base_character in characters_to_check:
        related_character = find_related_hanzi(base_character, component_to_chars, decompose_hanzi)
        if related_character:
            if related_character == base_character or related_character in decompose_hanzi(base_character):
                invalid_cases.append((base_character, related_character))
            else:
                results.append((base_character, related_character))
        else:
            invalid_cases.append((base_character, None))

    print(f"Processed {len(results)} valid base characters with related characters.")
    print(f"Found {len(invalid_cases)} invalid cases (overlap or no related character).")
    save_results_as_array(results)
    print("Results saved in the output directory.")