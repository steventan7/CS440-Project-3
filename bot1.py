'''
Implementation for Bot1
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

import numpy as np
import random
from colorama import init, Back, Style
init(autoreset=True)

D = 12


# Used to visualize the difference components of the game
def visualize_grid (image):
    for i in range(D):
        curr_row = ""
        for j in range(D):
            if image[i][j] == 'R':
                curr_row += (Style.RESET_ALL + Back.RED + "R_")
            elif image[i][j] == 'B':
                curr_row += (Style.RESET_ALL + Back.BLUE + "B_")
            elif image[i][j] == 'Y':
                curr_row += (Style.RESET_ALL + Back.YELLOW + "Y_")
            elif image[i][j] == 'G':
                curr_row += (Style.RESET_ALL + Back.GREEN  + "G_")
            else:
                curr_row += (Style.RESET_ALL + Back.WHITE + "__")
        print(curr_row)
    print()
    print()


'''
Creates a 20X20 image each tile with color {Red, Blue, Yellow, Green} based upon the specified implementation
'''
def create_image():
    rows = set([r for r in range(D)])
    cols = set([c for c in range(D)])
    image = np.full((D, D), "", dtype=str)

    pos = 'R'
    is_dangerous = False
    while rows and cols:
        colors = ['R', 'Y', 'B', 'G']
        for i in range(4):
            color = random.choice(list(colors))
            colors.remove(color)
            if pos == 'R':
                row = random.choice(list(rows))
                rows.remove(row)
                image[row] = color
            else:
                col = random.choice(list(cols))
                cols.remove(col)
                image[:, col] = color
            pos = 'C' if pos == 'R' else 'R'
        coin_flip = random.random()
        pos = 'R' if coin_flip <= .5 else 'C'
    return image, is_dangerous


if __name__ == '__main__':
    dataset = {}
    x = 0
    for i in range(1):
        image, dangerous = create_image()
        dataset[tuple(map(tuple, image))] = dangerous
        if dangerous:
            visualize_grid(image)
            x += 1
    print(x)
