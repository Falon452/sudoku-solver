import kivy
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import numpy as np
from minizinc import Instance, Model, Solver

kivy.require('1.9.1')


class StartScreen(Screen):
    def start_program(self):
        self.manager.current = 'image'


class ImageScreen(Screen):

    def upload(self):
        self.img.source = self.path.text

    def solve(self):
        tmp = self.img.source
        self.img.source = solver(tmp)


class InterfaceApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(StartScreen(name='start'))
        sm.add_widget(ImageScreen(name='image'))

        return sm


def solver(image_path):

    image = cv2.imread(image_path)
    cv2.imshow("Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Image", blur)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    cv2.imshow("Image", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    left_upper = []
    right_upper = []
    left_lower = []
    right_lower = []
    left_upper_sum = 1000000000
    right_upper_sum = 0
    left_lower_sum = 0
    right_lower_sum = 0

    for i in best_cnt:
        x, y = i[0][0], i[0][1]
        if x + y > right_lower_sum:
            right_lower_sum = x + y
            right_lower = [x, y]
        if x + y < left_upper_sum:
            left_upper_sum = x + y
            left_upper = [x, y]
        if x - y < left_lower_sum:
            left_lower_sum = x - y
            left_lower = [x, y]
        if y - x < right_upper_sum:
            right_upper_sum = y - x
            right_upper = [x, y]

    width, height = 400, 400
    pts1 = np.float32([left_upper, right_upper, left_lower, right_lower])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarped = cv2.warpPerspective(image, matrix, (width, height))
    imgWarped = ~imgWarped  # invert colors

    step = width // 9
    sudoku = [[None for _ in range(9)] for _ in range(9)]

    for i in range(9):
        i_err = 3
        if i > 2:
            i_err = 4
        if i > 5:
            i_err = 6
        for j in range(9):
            sudoku[i][j] = imgWarped[0 + i * step + i_err: (i + 1) * step, 0 + j * step + i_err: (j + 1) * step]
            cv2.imshow(f"cell {i + 1} {j + 1}",
                       imgWarped[0 + i * step + i_err: (i + 1) * step, 0 + j * step + i_err: (j + 1) * step])

    # TENSORFLOW



    # MINIZINC

    sudoku_solver = Model("./sudoku-solver.mzn")
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, sudoku_solver)
    instance['board'] = [[0, 0, 4, 0, 5, 0, 0, 0, 0],
                         [9, 0, 0, 7, 3, 4, 6, 0, 0],
                         [0, 0, 3, 0, 2, 1, 0, 4, 9],
                         [0, 3, 5, 0, 9, 0, 4, 8, 0],
                         [0, 9, 0, 0, 0, 0, 0, 3, 0],
                         [0, 7, 6, 0, 1, 0, 9, 2, 0],
                         [3, 1, 0, 9, 7, 0, 2, 0, 0],
                         [0, 0, 9, 1, 8, 2, 0, 0, 3],
                         [0, 0, 0, 0, 6, 0, 1, 0, 0]]

    result = instance.solve()
    solution = result["puzzle"]

    sudoku_clean = cv2.imread("sudoku_clean.png")
    sudoku_clean = cv2.resize(sudoku_clean, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 1
    for i in range(9):
        i_err = 0
        if i > 2:
            i_err = 2
        if i > 5:
            i_err = 4
        for j in range(9):
            pos = (14 + i * step + i_err, 32 + j * step + i_err)
            sudoku_clean = cv2.putText(sudoku_clean, str(solution[j][i]), pos, font,
                                       fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Final Image", sudoku_clean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sudoku_clean


if __name__ == '__main__':
    InterfaceApp().run()