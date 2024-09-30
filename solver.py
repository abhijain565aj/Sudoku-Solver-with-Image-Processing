# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # # import tesseract


# # def get_sudoku_from_image(image_path):
# #     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # grayscale conversion
# #     # noise reduction with gaussian blur
# #     img = cv2.GaussianBlur(img, (5, 5), 0)
# #     # detect the largest contour in the image
# #     img = cv2.adaptiveThreshold(
# #         img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# #     contours, _ = cv2.findContours(
# #         img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
# #     largest_contour = contours[0]
# #     # get the corners of the largest contour
# #     peri = cv2.arcLength(largest_contour, True)
# #     approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
# #     # transform and crop the image
# #     pts = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
# #     height = width = 450
# #     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# #     matrix = cv2.getPerspectiveTransform(pts, pts2)
# #     img = cv2.warpPerspective(img, matrix, (width, height))
# #     # save the image
# #     cv2.imwrite('sudoku.jpg', img)
# #     return img
# # import cv2
# # import numpy as np


# # def order_points(pts):
# #     rect = np.zeros((4, 2), dtype="float32")
# #     s = pts.sum(axis=1)
# #     rect[0] = pts[np.argmin(s)]  # top-left
# #     rect[2] = pts[np.argmax(s)]  # bottom-right
# #     diff = np.diff(pts, axis=1)
# #     rect[1] = pts[np.argmin(diff)]  # top-right
# #     rect[3] = pts[np.argmax(diff)]  # bottom-left
# #     return rect


# # def get_sudoku_from_image(image_path):
# #     # Load the image in grayscale
# #     img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# #     if img_original is None:
# #         raise ValueError("Image not found or could not be loaded.")

# #     # Noise reduction with Gaussian blur
# #     img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)

# #     # Thresholding to binary (used for contour detection)
# #     img_thresh = cv2.adaptiveThreshold(
# #         img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# #     # Detect the largest contour in the image (Sudoku grid)
# #     contours, _ = cv2.findContours(
# #         img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     if not contours:
# #         raise ValueError("No contours found in the image.")

# #     # Sort contours by area and get the largest one
# #     print(contours)
# #     contours = sorted(contours, key=cv2.contourArea, reverse=True)
# #     largest_contour = contours[0]

# #     # Get the corners of the largest contour (the grid)
# #     peri = cv2.arcLength(largest_contour, True)
# #     approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

# #     # Ensure the largest contour has exactly 4 points (for a square grid)
# #     if len(approx) != 4:
# #         raise ValueError("The largest contour does not have 4 corners.")

# #     # Order the points in top-left, top-right, bottom-left, bottom-right order
# #     pts = order_points(approx.reshape(4, 2))
# #     print(pts)
# #     # Define the target size of the Sudoku grid
# #     height = width = 450
# #     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# #     print(pts2)

# #     # Get the perspective transformation matrix
# #     matrix = cv2.getPerspectiveTransform(pts, pts2)
# #     print(matrix)

# #     # Warp the original grayscale image using the perspective matrix
# #     img_warped = cv2.warpPerspective(img_original, matrix, (width, height))

# #     # Save and return the final transformed image
# #     cv2.imwrite('sudoku.jpg', img_thresh)
# #     return img_warped
# import cv2
# import numpy as np


# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]  # top-left
#     rect[2] = pts[np.argmax(s)]  # bottom-right
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]  # top-right
#     rect[3] = pts[np.argmax(diff)]  # bottom-left
#     return rect


# def get_sudoku_from_image(image_path):
#     # Load the image in grayscale
#     img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img_original is None:
#         raise ValueError("Image not found or could not be loaded.")

#     # Noise reduction with Gaussian blur
#     img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)

#     # Thresholding to binary (used for contour detection)
#     img_thresh = cv2.adaptiveThreshold(
#         img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     # DEBUG: Visualize the thresholded image to verify it
#     cv2.imwrite('debug_thresh.jpg', img_thresh)

#     # Detect the largest contour in the image (Sudoku grid)
#     contours, _ = cv2.findContours(
#         img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("No contours found in the image.")

#     # Sort contours by area and get the largest one
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     print(contours)
#     largest_contour = contours[0]

#     # DEBUG: Draw the largest contour on a copy of the original image
#     img_contours = cv2.drawContours(
#         img_original.copy(), [largest_contour], -1, (0, 255, 0), 3)
#     cv2.imwrite('debug_contours.jpg', img_contours)

#     # Get the corners of the largest contour (the grid)
#     peri = cv2.arcLength(largest_contour, True)
#     approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

#     # Ensure the largest contour has exactly 4 points (for a square grid)
#     if len(approx) != 4:
#         raise ValueError("The largest contour does not have 4 corners.")

#     print(approx)
#     # DEBUG: Draw the approximated polygon on the original image
#     img_approx = cv2.polylines(img_original.copy(), [
#                                approx], True, (255, 0, 0), 10)
#     cv2.imwrite('debug_approx.jpg', img_approx)

#     # Order the points in top-left, top-right, bottom-left, bottom-right order
#     pts = order_points(approx.reshape(4, 2))

#     # Define the target size of the Sudoku grid
#     height = width = 450
#     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

#     # Get the perspective transformation matrix
#     matrix = cv2.getPerspectiveTransform(pts, pts2)

#     # Warp the original grayscale image using the perspective matrix
#     img_warped = cv2.warpPerspective(img_original, matrix, (width, height))

#     # DEBUG: Save the warped perspective image
#     cv2.imwrite('debug_warped.jpg', img_warped)

#     # Save and return the final transformed image
#     cv2.imwrite('sudoku.jpg', img_warped)
#     return img_warped

# import cv2
# import numpy as np


# def get_sudoku_from_image(image_path):
#     # Load the image in grayscale
#     img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img_original is None:
#         raise ValueError("Image not found or could not be loaded.")

#     # Noise reduction with Gaussian blur
#     img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)

#     # Thresholding to binary (used for contour detection)
#     img_thresh = cv2.adaptiveThreshold(
#         img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#     # DEBUG: Visualize the thresholded image to verify it
#     cv2.imwrite('debug_thresh.jpg', img_thresh)

#     # Detect the contours
#     contours, _ = cv2.findContours(
#         img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise ValueError("No contours found in the image.")

#     # Sort contours by area and get the largest one
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     print(contours)
#     # plot all contours on the image
#     img1 = img_thresh.copy()
#     for contour in contours:
#         img1 = cv2.drawContours(img1, contour, -1, (0, 255, 0), 3)

#     cv2.imwrite('debug.jpg', img1)

#     # Filter contours by area to eliminate small irrelevant contours
#     MIN_CONTOUR_AREA = 1000  # Adjust this value as needed based on the grid size
#     largest_contour = None
#     for contour in contours:
#         if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
#             largest_contour = contour
#             break

#     if largest_contour is None:
#         raise ValueError("No valid Sudoku grid found.")

#     # DEBUG: Draw the largest contour on a copy of the original image
#     img_contours = cv2.drawContours(
#         img_original.copy(), [largest_contour], -1, (0, 255, 0), 3)
#     cv2.imwrite('debug_contours.jpg', img_contours)

#     # Get the corners of the largest contour (the grid)
#     peri = cv2.arcLength(largest_contour, True)
#     approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

#     # Ensure the largest contour has exactly 4 points (for a square grid)
#     if len(approx) != 4:
#         raise ValueError(
#             f"The largest contour does not have 4 corners. It has {len(approx)} corners.")

#     # Order the points in top-left, top-right, bottom-left, bottom-right order
#     pts = np.float32([pt[0] for pt in approx])

#     # Define the target size of the Sudoku grid
#     height = width = 450
#     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

#     # Get the perspective transformation matrix
#     matrix = cv2.getPerspectiveTransform(pts, pts2)

#     # Warp the original grayscale image using the perspective matrix
#     img_warped = cv2.warpPerspective(img_original, matrix, (width, height))

#     # DEBUG: Save the warped perspective image
#     cv2.imwrite('debug_warped.jpg', img_warped)

#     # Save and return the final transformed image
#     cv2.imwrite('sudoku.jpg', img_warped)
#     return img_warped

import cv2
import numpy as np
# import pytesseract
import tesserocr
from PIL import Image


# Adjust the path to your installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def get_sudoku_from_image(image_path):
    # Load the image in grayscale
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError("Image not found or could not be loaded.")

    # Noise reduction with Gaussian blur
    img_blur = cv2.GaussianBlur(img_original, (5, 5), 0)

    # Apply Canny edge detection to highlight edges
    img_edges = cv2.Canny(img_blur, 50, 150)

    # DEBUG: Save the edge-detected image to visualize it
    cv2.imwrite('debug_edges.jpg', img_edges)

    # Apply dilation to connect broken lines of the grid
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_edges, kernel, iterations=1)

    # DEBUG: Save the dilated image to visualize it
    cv2.imwrite('debug_dilated.jpg', img_dilated)

    # Detect the contours after dilation
    contours, hierarchy = cv2.findContours(
        img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")

    # Sort contours by area and get the largest one (presumably the Sudoku grid)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter out very small contours
    MIN_CONTOUR_AREA = 1000  # Adjust based on the image size
    largest_contour = None
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            largest_contour = contour
            break

    if largest_contour is None:
        raise ValueError("No valid Sudoku grid found.")
    print(largest_contour)
    # DEBUG: Draw the largest contour on a copy of the original image
    img_contours = cv2.drawContours(
        img_original.copy(), [largest_contour], -1, (0, 255, 0), 3)
    cv2.imwrite('debug_contours.jpg', img_contours)

    # Get the corners of the largest contour (the grid)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

    # Ensure the largest contour has exactly 4 points (for a square grid)
    if len(approx) != 4:
        raise ValueError(
            f"The largest contour does not have 4 corners. It has {len(approx)} corners.")
    print(approx)
    # Order the points in top-left, top-right, bottom-left, bottom-right order
    pts = np.float32([pt[0] for pt in approx])
    pts = order_points(pts)
    print(pts)
    # Define the target size of the Sudoku grid
    height = width = 512
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts, pts2)

    # Warp the original grayscale image using the perspective matrix
    img_warped = cv2.warpPerspective(img_original, matrix, (width, height))

    # DEBUG: Save the warped perspective image
    cv2.imwrite('debug_warped.jpg', img_warped)

    # Save and return the final transformed image
    cv2.imwrite('sudoku.jpg', img_warped)
    return img_warped


def divide_into_81_cells(img):
    cells = []
    cell_size = img.shape[0] // 9
    for i in range(9):
        for j in range(9):
            cell = img[i*cell_size:(i+1)*cell_size, j *
                       cell_size:(j+1)*cell_size]
            cells.append(cell)
    return cells


def identify_number(img, ind):
    # Resize the image to 28x28
    img = cv2.resize(img, (64, 64))
    # crop the image by 10% from all side to remove the border
    # img = img[8:56, 8:56]

    # delete border non-thick lines
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    # Apply Gaussian blur to the image
    # thresholding
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert the colors (if needed)
    cv2.imwrite('cells/cell'+str(ind)+'.jpg', img)
    img = cv2.bitwise_not(img)
    # save the image to visualize

    # Ensure the image is 8-bit unsigned integers
    img = (img * 255).astype(np.uint8)

    # Convert NumPy array to a PIL image
    pil_img = Image.fromarray(img)

    # Use tesserocr to extract text
    with tesserocr.PyTessBaseAPI() as api:
        api.SetImage(pil_img)
        api.SetVariable("tessedit_char_whitelist",
                        " 123456789")  # Restrict to digits
        text = api.GetUTF8Text()
    return text.strip()


def sudoku(img):
    img = get_sudoku_from_image(img)
    cells = divide_into_81_cells(img)
    # numpy array to store the sudoku
    sudoku = list([['.' for i in range(9)] for j in range(9)])
    for i, cell in enumerate(cells):
        number = identify_number(cell, i)
        row = i // 9
        col = i % 9
        sudoku[row][col] = number

    return sudoku


ans = sudoku('sudoku_1.jpg')
for i in ans:
    for j in i:
        print(j, end=' ')
    print()
