import pathlib

import cv2
import numpy as np
from fastai.learner import load_learner
from minizinc import Instance, Model, Solver
from fastbook import *


def main():
    image = cv2.imread("sudoku.jpg")
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
    best_cnt = contours[0]
    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    right_upper, left_lower, right_lower, left_upper = get_corners(best_cnt)

    width, height = 400, 400
    pts1 = np.float32([left_upper, right_upper, left_lower, right_lower])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarped = cv2.warpPerspective(image, matrix, (width, height))
    imgWarped = ~imgWarped  # invert colors
    results = get_sudoku_tiles(imgWarped, width)
    # solve(results, width, height)


def solve(results, width, height):
    sudoku_solver = Model("./sudoku-solver.mzn")
    gecode = Solver.lookup("gecode")
    instance = Instance(gecode, sudoku_solver)
    instance['board'] = results

    result = instance.solve()
    solution = result["puzzle"]

    sudoku_clean = cv2.imread("sudoku_clean.png")
    sudoku_clean = cv2.resize(sudoku_clean, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    step = width // 9
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


def get_corners(best_cnt):
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

    return right_upper, left_lower, right_lower, left_upper

def get_sudoku_tiles(img, width):
    step = width // 9
    sudoku = [[None for _ in range(9)] for _ in range(9)]

    for i in range(9):
        basic_err = 5
        i_err = basic_err
        if i > 2:
            i_err = 6
        if i > 5:
            i_err = 7
        for j in range(9):
            sudoku[i][j] = img[0 + i * step + i_err: (i + 1) * step + i_err - 2 * basic_err,
                           0 + j * step + i_err: (j + 1) * step + i_err - 2 * basic_err]
            cv2.imshow(f"cell {i + 1} {j + 1}",
                       img[0 + i * step + i_err: (i + 1) * step, 0 + j * step + i_err: (j + 1) * step])

    path = Path()

    pathlib.PosixPath = pathlib.WindowsPath

    learn_inf = load_learner(path / 'nn_blank_images.pkl')

    results = [[0 for _ in range(9)] for _ in range(9)]

    for i in range(9):
        for j in range(9):
            pred, _, probs = learn_inf.predict(sudoku[i][j])
            pred = int(pred)
            results[i][j] = pred

    return results


main()