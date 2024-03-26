from PIL import Image
import cv2
import numpy as np

char_map = {
    '      ': '⠄', '.     ': '⠁', '  .   ': '⠂', '. .   ': '⠃', '    . ': '⠄', '.   . ': '⠅', '  . . ': '⠆',
    '. . . ': '⠇', ' .    ': '⠈', '..    ': '⠉', ' ..   ': '⠊', '...   ': '⠋', ' .  . ': '⠌', '..  . ': '⠍',
    ' .. . ': '⠎', '... . ': '⠏', '   .  ': '⠐', '.  .  ': '⠑', '  ..  ': '⠒', '. ..  ': '⠓', '   .. ': '⠔',
    '.  .. ': '⠕', '  ... ': '⠖', '. ... ': '⠗', ' . .  ': '⠘', '.. .  ': '⠙', ' ...  ': '⠚', '....  ': '⠛',
    ' . .. ': '⠜', '.. .. ': '⠝', ' .... ': '⠞', '..... ': '⠟', '     .': '⠠', '.    .': '⠡', '  .  .': '⠢',
    '. .  .': '⠣', '    ..': '⠤', '.   ..': '⠥', '  . ..': '⠦', '. . ..': '⠧', ' .   .': '⠨', '..   .': '⠩',
    ' ..  .': '⠪', '...  .': '⠫', ' .  ..': '⠬', '..  ..': '⠭', ' .. ..': '⠮', '... ..': '⠯', '   . .': '⠰',
    '.  . .': '⠱', '  .. .': '⠲', '. .. .': '⠳', '   ...': '⠴', '.  ...': '⠵', '  ....': '⠶', '. ....': '⠷',
    ' . . .': '⠸', '.. . .': '⠹', ' ... .': '⠺', '.... .': '⠻', ' . ...': '⠼', '.. ...': '⠽', ' .....': '⠾',
    '......': '⠿'
}


def match_char(pixels):
    string = ""
    for i in pixels:
        if i > 0:
            string += ' '
        else:
            string += '.'

    return char_map.get(string)


def prep_image(name, size):
    og = Image.open(name)
    gray = og.convert('L')
    gray = gray.resize((int(gray.size[0] * size), int(gray.size[1] * size)), Image.Resampling.LANCZOS)

    avg_color = 0
    for i in range(gray.height):
        for j in range(gray.width):
            avg_color += gray.getpixel((j, i))

    avg_color /= (gray.height * gray.width)
    bw_func = lambda x: 255 if x > avg_color else 0
    bw = gray.convert('L').point(bw_func, mode='1')
    y_count = bw.height // 3
    x_count = bw.width // 2
    return [bw, x_count, y_count]


def prep_image_edges(name, size):
    og = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    og = cv2.flip(og, 0)
    og = cv2.rotate(og, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.resize(og, (int(og.shape[0] * size), int(og.shape[1] * size * 2)))

    edges = cv2.Canny(gray, 255 / 3, 255)

    coords = np.where(edges != [0])
    coords_set = set(zip(coords[0], coords[1]))

    y_count = gray.shape[1] // 3
    x_count = gray.shape[0] // 2
    coords_dict = {}
    for i in range(gray.shape[1]):
        for j in range(gray.shape[0]):
            if (j, i) not in coords_set:
                coords_dict[(j, i)] = 1
            else:
                coords_dict[(j, i)] = 0

    return [coords_dict, x_count, y_count]


def generate(BW, x_count, y_count):
    lines = []
    for i in range(y_count):
        line = ''
        for j in range(x_count):
            crop_rect = (j * 2, i * 3, (j + 1) * 2, (i + 1) * 3)
            cropped = BW.crop(crop_rect)
            area = [cropped.getpixel((0, 0)), cropped.getpixel((1, 0)), cropped.getpixel((0, 1)),
                    cropped.getpixel((1, 1)), cropped.getpixel((0, 2)), cropped.getpixel((1, 2))]
            line += match_char(area)
        line += '\n'
        lines.append(line)

    return lines


def generate_edges(coords, x_count, y_count):
    lines = []
    for i in range(y_count):
        line = ''
        for j in range(x_count):
            x = j * 2
            # xMax = (j+1) * xCount
            y = i * 3
            # yMax = (i+1) * yCount
            area = [coords.get((x, y)), coords.get((x + 1, y)), coords.get((x, y + 1)),
                    coords.get((x + 1, y + 1)), coords.get((x, y + 2)), coords.get((x + 1, y + 2))]

            line += match_char(area)
        line += '\n'
        lines.append(line)
    return lines
