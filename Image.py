'''
Implementation for Image
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

import numpy as np
import random
from colorama import init, Back, Style
init(autoreset=True)

D = 20


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


class Image:
    def __init__(self):
        self.pixels = np.full((D, D), "", dtype=str)
        self.is_dangerous = False
        self.dangerous_color = "X"

    '''
    Creates a 20X20 image each tile with color {Red, Blue, Yellow, Green} based upon the specified implementation
    '''
    def create_image(self):
        rows = set([r for r in range(D)])
        cols = set([c for c in range(D)])

        pos, self.is_dangerous = 'ROW', False
        while rows and cols:
            prev_color, prev_pos = '', ''
            colors = ['R', 'Y', 'B', 'G']
            for i in range(4):
                color = random.choice(list(colors))
                colors.remove(color)
                if pos == 'ROW':
                    row = random.choice(list(rows))
                    rows.remove(row)
                    self.pixels[row] = color
                    if prev_color == 'R' and color == 'Y' and prev_pos == 'COL':
                        # print("DANGER")
                        self.is_dangerous = True
                        self.dangerous_color = 'Y' # HOW TO DETERMINE?
                else:
                    col = random.choice(list(cols))
                    cols.remove(col)
                    self.pixels[:, col] = color
                    if prev_color == 'R' and color == 'Y' and prev_pos == 'ROW':
                        # print("DANGER")
                        self.is_dangerous = True
                        self.dangerous_color = 'Y' # HOW TO DETERMINE?

                prev_pos = pos
                pos = 'COL' if pos == 'ROW' else 'ROW'
                prev_color = color
                # visualize_grid(self.pixels)
                # print(color)
                # x = input()
                # if self.is_dangerous:
                #     return self.pixels, self.is_dangerous
            # print("cycle completed")
            coin_flip = random.random()
            pos = 'ROW' if coin_flip <= .5 else 'COL'
        return self.pixels, self.is_dangerous, self.dangerous_color
