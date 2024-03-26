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
    gray = og.convert('L', dither=Image.Dither.FLOYDSTEINBERG)
    gray = gray.resize((int(gray.width * size), int(gray.height * size)), Image.Resampling.LANCZOS)
    avg_color = np.sum(gray) / (gray.width * gray.height)
    bw_func = lambda x: 255 if x > avg_color else 0
    bw = gray.convert('L').point(bw_func, mode='1')
    y_count = bw.height // 3
    x_count = bw.width // 2
    return [bw, x_count, y_count]


def prep_image_edges(name, size):
    og = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    og = cv2.flip(og, 0)
    og = cv2.rotate(og, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.resize(og, (int(og.shape[0] * size), int(og.shape[1] * size * 3)))
    edges = cv2.Canny(gray, 255 / 3, 255)
    coords = np.where(edges != [0])
    coords_set = set(zip(coords[0], coords[1]))
    y_count = gray.shape[1] // 3
    x_count = gray.shape[0] // 2
    edge_coords = np.ones((gray.shape[1], gray.shape[0]))
    for loc in coords_set:
        edge_coords[loc[1]][loc[0]] = 0

    return [edge_coords, x_count, y_count]


def generate(BW, x_count, y_count):
    lines = []
    coords = np.array(BW)
    for i in range(y_count):
        line = ''
        y = i * 3
        for j in range(x_count):
            x = j * 2
            area = [coords[y, x], coords[y, x + 1],
                    coords[y + 1, x], coords[y + 1, x + 1],
                    coords[y + 2 , x], coords[y + 2, x + 1]]
            line += match_char(area)
        line += '\n'
        lines.append(line)

    return lines


def generate_edges(coords, x_count, y_count):
    lines = []
    for i in range(y_count):
        line = ''
        y = i * 3
        for j in range(x_count):
            x = j * 2
            area = [coords[y, x], coords[y, x + 1],
                    coords[y + 1, x], coords[y + 1, x + 1],
                    coords[y + 2 , x], coords[y + 2, x + 1]]

            line += match_char(area)
        line += '\n'
        lines.append(line)
    return lines
