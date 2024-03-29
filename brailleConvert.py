from PIL import Image
import cv2
import numpy as np
from collections import deque


"""
Dictionary mapping all possible configurations of 6 dots in a 3x2
space to a braille character

"""
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


def match_char(pixels, invert):
    string = ""
    for i in pixels:
        if i > 0:
            string += '.' if invert else ' '
        else:
            string += ' ' if invert else '.'

    return char_map.get(string)

#Preparing the image for the pixel thresholding approach
def prep_image(name, size):
    og = Image.open(name)
    gray = og.convert('L')

    """
    If the size argument is less than 1, it's used to directly scale the output.
    Since most images are many hundreds or thousands of pixels wide, it's unlikely that
    a scaling factor of more than 1 will be needed.

    if the size argument is more than 1, it's interpreted as how many characters wide
    the final output should be. 

    """
    size = size / (gray.width // 2) if size > 1 else size
    gray = gray.resize((int(gray.width * size), int(gray.height * size)), Image.Resampling.LANCZOS)
    """
    Finds average grayscale value of all the pixels, then uses it as a threshold to
    determine which pixels are black or white in the final binary image
    """
    avg_color = np.sum(gray) / (gray.width * gray.height)
    bw_func = lambda x: 255 if x > avg_color else 0
    bw = gray.convert('L').point(bw_func, mode='1')
    y_count = bw.height // 3
    x_count = bw.width // 2
    return [bw, x_count, y_count]

#Performing the Canny edge detection for the Canny edge detection approach
def prep_image_edges(name, size):
    og = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    size = size / (og.shape[1] // 2) if size > 1 else size
    gray = cv2.resize(og, (int(og.shape[1] * size), int(og.shape[0] * size)))
    """
    Performs Canny edge detection on the grayscale image, then uses a NumPy array
    to represent all the pixels in the grayscale image as either a 1 or 0,
    with 0 representing a pixel detected as an edge
    """
    edges = cv2.Canny(gray, 255 / 3, 255)
    coords = np.where(edges != [0])
    coords_set = set(zip(coords[0], coords[1]))
    y_count = gray.shape[0] // 3
    x_count = gray.shape[1] // 2
    edge_coords = np.ones((gray.shape[0], gray.shape[1]))
    for loc in coords_set:
        edge_coords[loc[0]][loc[1]] = 0

    return [edge_coords, x_count, y_count]


def generate(BW, x_count, y_count, invert):
    """
    Iterates over each 3x2 area of pixels in the binary image, using the above
    dictionary to match it to a unique braille character
    """
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
            line += match_char(area, invert)
        line += '\n'
        lines.append(line)

    return lines


def generate_edges(coords, x_count, y_count, invert):
    lines = []
    for i in range(y_count):
        line = ''
        y = i * 3
        for j in range(x_count):
            x = j * 2
            area = [coords[y, x], coords[y, x + 1],
                    coords[y + 1, x], coords[y + 1, x + 1],
                    coords[y + 2 , x], coords[y + 2, x + 1]]

            line += match_char(area, invert)
        line += '\n'
        lines.append(line)
    return lines

