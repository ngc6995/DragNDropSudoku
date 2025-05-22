import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(image: np.ndarray, size: tuple[int,int]):
    plt.figure(figsize=size)
    if len(image.shape) > 2:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.axis(False)
    plt.show()

def demonstrate(*images, titles: list, size: tuple[int,int]):
    fig = plt.figure(figsize=size)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i+1)
        if titles:
            plt.title(titles[i])
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis(False)
    plt.show()

def draw_cells(cells: np.ndarray, size: tuple[int,int]):
    i = 1
    fig = plt.figure(figsize=size)
    for row in range(9):
        for col in range(9):
            fig.add_subplot(9, 9, i)
            plt.imshow(cells[row, col], cmap='gray')
            plt.axis(False);
            i += 1
    plt.show()

def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    # Function to calculate the angle of a point with respect to the centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    # Sort points based on the angle
    sorted_points = sorted(points, key=angle_from_centroid)
    sorted_points = np.array(sorted_points, dtype=np.float32)
    return sorted_points

def extract_grid(sudoku_image: np.ndarray, size: int, pct1: float=10.0, pct2: float=0.01) -> np.ndarray:
    # The dimension of output sudoku grid image is (size, size, 3), size must be multiple of 9, 9*28, 9*64 ...
    # Preprocess image
    image_gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    #edge = cv2.Canny(blur, 50, 100)
    #contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_blurred = cv2.bilateralFilter(image_gray, 5, 50, 50)
    image_threshed = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    contours, hierarchy = cv2.findContours(image_threshed
                                           , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Contours filter
    candidates = []
    image_area = sudoku_image.shape[0] * sudoku_image.shape[1]
    for contour_id, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        # It's a quadrilateral and area greater then pct1% of image?
        if len(approx_polygon) == 4 and area > pct1 * image_area / 100:
            # Initialize candidate score to 0
            candidates.append([contour_id, approx_polygon, 0])
    if len(candidates) > 0:
        # Calculate scores of each candidate
        for candidate in candidates:
            contour_id = candidate[0]
            # How many child contours under candidate(parent is this candidate)?
            child_contour_ids = np.where(hierarchy[0,:,3]==contour_id)
            for child_contour_id in child_contour_ids[0]:
                child_contour = contours[child_contour_id]
                perimeter = cv2.arcLength(child_contour, True)
                approx_polygon = cv2.approxPolyDP(child_contour, 0.02 * perimeter, True)
                area = cv2.contourArea(child_contour)
                # It's a quadrilateral and area greater then pct2% of image?
                if len(approx_polygon) == 4 and area > pct2 * image_area / 100:
                    candidate[2] += 1  # Add 1 to candidate's score
        # A contour has max quadrilateral child contours, most probable is a sudoku grid
        best_candidate = max(candidates, key=lambda x:x[2])
        # Candidate's score is equal to 0?
        if best_candidate[2] > 0:
            approx_polygon = best_candidate[1]
            points = approx_polygon.reshape((4, 2))
            sorted_points = sort_points_clockwise(points)
            destination = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src=sorted_points, dst=destination)
            grid_image = cv2.warpPerspective(sudoku_image, M, (size, size))
            return grid_image
        else:
            return None
    else:
        return None
    
def extract_cells(grid_image: np.ndarray) -> np.ndarray:
    # Only grayscale images will be used as input to the network
    if len(grid_image.shape) > 2:
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
    grid_image_height, grid_image_width = grid_image.shape
    # Split height to 9 cells
    cell_height = grid_image_height // 9
    # Split width to 9 cells
    cell_width = grid_image_width // 9
    # Total 81 cell images
    cells = np.zeros((9, 9, cell_height, cell_width), dtype=np.uint8)
    for row in range(9):
        for col in range(9):
            r1 = cell_height * row
            r2 = r1 + cell_height
            c1 = cell_width * col
            c2 = c1 + cell_width
            cell = grid_image[r1:r2, c1:c2]
            cells[row, col] = cell
    return cells

def is_empty_cell(cell_image: np.ndarray, crop_ratio: float=0.4) -> bool:
    height, width = cell_image.shape
    crop_size = round((width - width * crop_ratio) / 2)
    # Crop the center part of an cell image
    image_cropped = cell_image[crop_size:height-crop_size, crop_size:width-crop_size]
    #image_blurred = cv2.GaussianBlur(image_cropped, (7,7), 1)
    image_blurred = cv2.blur(image_cropped, (7,7))
    image_edged = cv2.Canny(image_blurred, 50, 100)
    # if no edge found, the color of the cropped image is black, it's an empty cell
    # This may not work 100% correctly, but there is not enough images to train the
    # network to recognize empty cells
    return np.all(image_edged==0)

# Find out how many numbers can't be filled in a cell
def find_constraints(grid: np.ndarray, row: int, col: int) -> int:
    row_nums = grid[row]
    col_nums = grid[:, col]
    sub_grid_row, sub_grid_col = 3 * (row // 3), 3 * (col // 3)
    sub_grid_nums = grid[sub_grid_row:sub_grid_row+3, sub_grid_col:sub_grid_col+3].flatten()
    combined = np.concatenate((row_nums, col_nums, sub_grid_nums))
    combined = combined[combined!=0]
    unique_combined = np.unique(combined)
    return len(unique_combined)

def find_empty_cell(grid: np.ndarray):
    constraints = []
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0:
                constraints.append((row, col, find_constraints(grid, row, col)))
    if len(constraints) > 0:
        # Choose the most constrained cell
        row, col, _ = max(constraints, key=lambda x:x[2])
        return row, col
    else:
        return False

def is_valid(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    # Check row
    if num in grid[row]:
        return False
    # Check column
    if num in grid[:, col]:
        return False
    sub_grid_row, sub_grid_col = 3 * (row // 3), 3 * (col // 3)
    # Check sub grid
    if num in grid[sub_grid_row:sub_grid_row+3, sub_grid_col:sub_grid_col+3]:
        return False
    return True

def solve(grid: np.ndarray):
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        # Sudoku solved
        return True
    row, col = empty_cell
    for num in range(1, 10):
        if is_valid(grid, row, col, num):
            grid[row, col] = num
            if solve(grid):
                return True
            # Backtrack
            grid[row, col] = 0
    # No solution
    return False

# Get the filled in numbers
def get_solution(puzzle_unsolved: np.ndarray, puzzle_solved: np.ndarray) -> list[tuple[int,int,int]]:
    solution = []
    empty_cells = np.where(puzzle_unsolved==0)
    for row, col in zip(empty_cells[0], empty_cells[1]):
        digit_filled = puzzle_solved[row, col]
        solution.append((row, col, digit_filled))
    return solution

# Draw the filled in numbers to empty cells
def draw_solution(grid_image: np.ndarray, solution: list[tuple[int,int,int]], color: tuple[int,int,int]):
    grid_height, grid_width = grid_image.shape[0], grid_image.shape[1]
    cell_height, cell_width = grid_height // 9,  grid_width // 9
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.02 * cell_height
    thickness = 2 if fontScale > 0.6 else 1
    for row, col, digit in solution:
        # Get the size of the text
        (digit_width, digit_height), baseline = cv2.getTextSize(str(digit), fontFace, fontScale, thickness)
        # Calculate the center position
        center_x = (cell_width - digit_width) // 2
        center_y = (cell_height + digit_height) // 2  # use '+', because origin is at the bottom-left
        # Define the bottom-left corner of the text
        org = (center_x + col*cell_width, center_y + row*cell_height)
        # Put the text on the image
        cv2.putText(grid_image, str(digit), org, fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    print("Model: sudoku.py for Sudoku image processing.")
    print("Please run solver.py")
