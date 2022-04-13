# Sudoku solver

### Description 
An app created for Bio-inspired Algorithms course on AGH UST.
Used techologies: `Python` with its libraries: `OpenCV` , `FastAI`, `Kivy` and `MiniZinc`.
#### Authors
- Damian Tworek - https://github.com/Falon452
- Bogusław Błachut - https://github.com/bblachut
- Zuzanna Furtak - https://github.com/zfurtak

### Functionality
The app designed for solving sudoku uploaded from a photo.
Grid Detection -> Digit Classification -> Geocode Solver -> Draw Solved Sudoku
From uploaded photo the model detects sudoku grid, then separates boxes from the grid.
In each box it recognises numbers or plain space.
Read grid is forwarded to MiniZinc gecode solver as an 2D array, 
solved using declarative programming and is returned as an array.
Returned array is converted to photo and given back to user.

### Demo 

![sudoku gif](https://user-images.githubusercontent.com/92310164/163208377-b1a333f7-f172-4b74-8254-f6a0d4a9199b.gif)
