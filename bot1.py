'''
Implementation for Bot1
@author Ajay Anand, Yashas Ravi, Steven Tan
'''

from colorama import init, Back, Style
from Image import Image

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


if __name__ == '__main__':
    dataset = {}
    x = 0
    for i in range(100):
        image = Image()
        image.create_image()
        dataset[image] = image.is_dangerous
        if image.is_dangerous:
            x += 1
    print(x)
