import numpy as np
import cv2
import os, sys
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import sudoku

def preprocess_image(cell_image: np.ndarray):
    # Resize to 28x28
    image_resized = cv2.resize(cell_image, (28, 28), interpolation=cv2.INTER_AREA)
    # Convert to float32 and scale to [0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    # Normalize with mean=0.5, std=0.5
    image_normalized = (image_float - 0.5) / 0.5
    # Reshape to [1, 1, 28, 28] for model input
    image_preprocessed = image_normalized.reshape(1, 1, 28, 28)
    return image_preprocessed

def predict(cell_image: np.ndarray, onnx_model_path: str) -> int:
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Preprocess the cell image
    image_input = preprocess_image(cell_image)
    # Load the ONNX model
    net = cv2.dnn.readNetFromONNX(onnx_model_path)
    # Set input for the network
    net.setInput(image_input)
    # Run inference
    output = net.forward()
    # Get predicted digit (index of max logit)
    predicted_class = np.argmax(output, axis=1)[0]
    predicted_digit = classes[predicted_class]
    return predicted_digit

def recognize_digits_onnx(grid_image: np.ndarray, onnx_model_path: str) -> np.ndarray:
    arr = np.zeros((9,9), dtype=np.uint8)
    cells = sudoku.extract_cells(grid_image)
    for row in range(9):
        for col in range(9):
            cell = cells[row, col]
            if not sudoku.is_empty_cell(cell):
                predicted_digit = predict(cell, onnx_model_path)
                arr[row, col] = predicted_digit
    return arr

def drop(event):
    # Get the dropped file path
    file_path = event.data
    # Remove braces (if file path contains space character, braces will be at the beginning and end)
    file_path = file_path.strip('{}')
    _, extension = os.path.splitext(file_path)
    root.title(f"Sudoku Solver - {os.path.basename(file_path)}")
    if extension in ['.bmp', '.jpg', '.png', '.webp']:
        image = cv2.imread(file_path)
        sudoku_grid = sudoku.extract_grid(image, size=9*50)
        if sudoku_grid is not None:
            puzzle = recognize_digits_onnx(sudoku_grid, './models/digits.onnx')
            puzzle_copy = puzzle.copy()
            if sudoku.solve(puzzle):
                solution = sudoku.get_solution(puzzle_copy, puzzle)
                sudoku.draw_solution(sudoku_grid, solution, (0,0,255))
                sudoku_grid = cv2.cvtColor(sudoku_grid, cv2.COLOR_BGR2RGB)
                sudoku_grid = Image.fromarray(sudoku_grid)
                photo = ImageTk.PhotoImage(sudoku_grid)
                # Update the label's image
                label.configure(image=photo)
                # Store reference to prevent garbage collection
                label.image = photo
            else:
                label.config(image='', text="Puzzle is unsolvable!")
        else:
            label.configure(image='', text="Can't extract sudoku grid!")
    else:
        label.configure(image='', text="File format not supported!")

# Create the main window
root = TkinterDnD.Tk()
root.title('Sudoku Solver')
root.geometry('450x450')
# Icon for app running on a Windows OS
if sys.platform.startswith('win'):
    import ctypes
    myappid = 'tkinter.python.solver'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    root.iconbitmap(r'icon.ico')
root.resizable(False, False)

# Create a label to display the solution image
label = tk.Label(root, text='Drag & Drop a sudoku image here', font=('Helvetica',15))
label.pack(expand=True, fill=tk.BOTH)

# Enable drag-and-drop functionality for files
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

# Start the Tkinter main loop
root.mainloop()
