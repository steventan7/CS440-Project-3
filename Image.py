'''
Implementation for Image
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

import numpy as np
import random
from colorama import init, Back, Style
init(autoreset=True)

D = 10


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
        self.pixels = np.full((D, D), 0, dtype=float)
        self.is_dangerous = 0
        self.third_wire = 0


    '''Creates a 20X20 image each tile with color {Red, Blue, Yellow, Green} based upon the specified implementation
    '''
    def create_image(self):
        rows = set([r for r in range(D)])
        cols = set([c for c in range(D)])

        self.is_dangerous = 0
        coin_flip = random.random()
        pos = 'ROW' if coin_flip <= .5 else 'COL'
        red_seen, red_pos = False, ''
        colors = [1, 2, 3, 4]
        
        for i in range(4):
            color = random.choice(list(colors))
            if color == 1:
                red_seen = True
                red_pos = pos
            colors.remove(color)
            if i == 3:
                self.third_wire = color
            if pos == 'ROW':
                row = random.choice(list(rows))
                rows.remove(row)
                self.pixels[row] = color
                if red_seen and red_pos == 'COL' and color == 3:
                    self.is_dangerous = 1
                if (i == 2):
                    self.third_wire = color
            else:
                col = random.choice(list(cols))
                cols.remove(col)
                self.pixels[:, col] = color
                if red_seen and red_pos == 'ROW' and color == 3:
                    self.is_dangerous = 1
                if (i == 2):
                    self.third_wire = color
                
            pos = 'COL' if pos == 'ROW' else 'ROW'
        targets = self.pixels.reshape(-1).astype(int)
        self.pixels = np.eye(5)[targets]
        self.pixels = self.pixels.flatten()
        return self.pixels, self.is_dangerous