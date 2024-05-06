import cv2
import numpy as np
import heapq

def generate_random_color():
    return list(np.random.randint(0, 255, size=3).tolist())

def mark_image(image, image_number):
    marked_image = image.copy()
    markers = np.zeros((image.shape[0], image.shape[1]), dtype=int)  # Store labels as integers
    drawing = False

    def mark_objects(event, x, y, flags, param):
        nonlocal current_label, drawing, random_color
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            # Draw the label on the visual image using a colored circle
            cv2.circle(marked_image, (x, y), 5, random_color, -1)
            markers[y, x] = current_label  # Direct assignment without using cv2.circle
        elif event == cv2.EVENT_MOUSEMOVE and drawing: # continue drawing
            cv2.circle(marked_image, (x, y), 5, random_color, -1)
            markers[y, x] = current_label  # Direct assignment without using cv2.circle
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False  # stop drawing
        elif event == cv2.EVENT_RBUTTONDOWN:
            current_label += 1
            random_color = generate_random_color()

    current_label = 1
    random_color = generate_random_color()
    cv2.namedWindow(f'Image {image_number}')
    cv2.setMouseCallback(f'Image {image_number}', mark_objects)

    while True:
        cv2.imshow(f'Image {image_number}', marked_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()
    cv2.imwrite(f'hw4/labeled_image_{image_number}.jpg', marked_image)
    return markers


def manual_grayscale(image):
    # Applying the formula: 0.3*R + 0.59*G + 0.11*B
    return (0.3 * image[:, :, 2] + 0.59 * image[:, :, 1] + 0.11 * image[:, :, 0]).astype(np.uint8)
   
def sobel_filter(img, direction='x'):
    if direction == 'x':
        # Sobel kernel for x direction
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        # Sobel kernel for y direction
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialize output image
    gradient = np.zeros_like(img, dtype=float)

    # Padding the image to handle border pixels
    img_padded = np.pad(img, ((1, 1), (1, 1)), mode='edge')

    # Convolution operation
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Element-wise multiplication and summation
            gradient[y, x] = np.sum(kernel * img_padded[y:y+3, x:x+3])

    return gradient

def compute_gradient_magnitude(image):
    gray_image = manual_grayscale(image)
    grad_x = sobel_filter(gray_image, 'x')
    grad_y = sobel_filter(gray_image, 'y')
    
    # Calculate the magnitude of gradients
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize the magnitude image to the range 0-255
    gradient_magnitude = np.clip((gradient_magnitude / gradient_magnitude.max()) * 255, 0, 255).astype(np.uint8)
    return gradient_magnitude

def initialize_markers_and_queue(image, markers):
    gradient_magnitude = compute_gradient_magnitude(image)
    priority_queue = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if markers[y, x] != 0:
                heapq.heappush(priority_queue, (gradient_magnitude[y, x], x, y))
    return priority_queue, gradient_magnitude

def meyers_flooding_algorithm(image, markers):
    gradient_magnitude = compute_gradient_magnitude(image)
    labels = np.zeros_like(image, dtype=int)  # All pixels initially unmarked (0)
    priority_queue = []

    # Initialize the queue and set initial markers
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if markers[y, x] != 0:
                labels[y, x] = markers[y, x]  # Set initial labels from markers
                heapq.heappush(priority_queue, (gradient_magnitude[y, x], x, y))
            else:
                labels[y, x] = 0  # Ensure all other pixels are initially unmarked

    # Process the queue
    while priority_queue:
        _, x, y = heapq.heappop(priority_queue)
        temp = []

        # Process neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                label_num = int(labels[ny, nx][0])
                if label_num == 0:  # Unprocessed
                    labels[ny, nx] = -2  # Mark as in queue
                    heapq.heappush(priority_queue, (gradient_magnitude[ny, nx], nx, ny))
                elif label_num == -2:  # Already in queue but not processed
                    continue
                elif label_num > 0:
                    if label_num not in temp:
                        temp.append(label_num)
        if len(temp) > 1:
            labels[y, x] = -1  # Mark as edge if conflicting labels
        elif len(temp) == 1:
            labels[y, x] = temp[0]
            
    return labels

def create_transparent_overlay(image, labels):
    # Check the dimensions of the image and labels to ensure compatibility
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel (color) image.")

    # Generate a color map
    unique_labels = np.unique(labels)
    label_colors = {label: generate_random_color() for label in unique_labels if label > 0}
    # print(label_colors)
    height, width, channels = image.shape
        

    for y in range(height):
        for x in range(width):
            label_num = labels[y, x][0]
            # print(label_num)
            if(label_num > 0):
                # print(label_colors[labels[y, x][0]])
                blue = image[y, x, 0]
                green = image[y, x, 1]
                red = image[y, x, 2]          
                image[y, x][0] = blue * 0.5 + label_colors[label_num][0] * 0.5
                image[y, x][1] = green * 0.5 + label_colors[label_num][1] * 0.5
                image[y, x][2] = red * 0.5 + label_colors[label_num][2] * 0.5
            else:
                image[y, x] = 0
            
    return image


image_paths = ['hw4/images/img1.jpg', 'hw4/images/img2.jpg', 'hw4/images/img3.jpg']
for index, path in enumerate(image_paths):
    image = cv2.imread(path)
    if image is None:
        print(f"Error loading image {path}")
        break
    
    markers = mark_image(image, index + 1)
    labels = meyers_flooding_algorithm(image, markers)
    # print(labels)

    # Create a semi-transparent overlay of the labels on the original image
    result_image = create_transparent_overlay(image, labels)

    # Display the results
    cv2.imshow(f'Segmented Image {index+1}', result_image)
    cv2.imwrite(f'hw4/results/img{index+1}_q1.jpg', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()