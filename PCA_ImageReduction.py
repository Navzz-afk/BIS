import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

# Generate a synthetic grayscale clean image (simple vertical gradient)
def generate_clean_image(size=(100, 100)):
    x = np.linspace(0, 255, size[1])
    y = np.linspace(0, 255, size[0])
    clean_image = np.outer(y, np.ones_like(x))
    return clean_image.astype(np.uint8)

# Add Gaussian noise to image
def add_noise(image, noise_level=25):
    noisy_image = image + np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# Get neighbors of a pixel (8-connected)
def get_neighbors(image, x, y):
    neighbors = []
    rows, cols = image.shape
    for i in range(max(0, x-1), min(rows, x+2)):
        for j in range(max(0, y-1), min(cols, y+2)):
            if i == x and j == y:
                continue
            neighbors.append(image[i, j])
    return np.array(neighbors)

# Update rule for a single cell/pixel
def update_cell(image, x, y, mutation_rate=0.1):
    neighbors = get_neighbors(image, x, y)
    current_value = image[x, y]
    
    # Weighted average of neighbors for smoothing
    new_value = np.mean(neighbors)
    
    # Mutation: small random noise to avoid local minima
    if np.random.rand() < mutation_rate:
        new_value += np.random.uniform(-10, 10)
        
    # Clamp value to valid grayscale range
    return np.clip(new_value, 0, 255)

# Parallel Cellular Algorithm with iteration visualization
def parallel_cellular_algorithm_visual(noisy_image, iterations=10, mutation_rate=0.1):
    image = noisy_image.copy().astype(np.float32)
    rows, cols = image.shape
    
    plt.figure(figsize=(5,5))
    for it in range(iterations):
        new_image = image.copy()
        for x in range(rows):
            for y in range(cols):
                new_image[x, y] = update_cell(image, x, y, mutation_rate)
        image = new_image
        
        # Visualize current iteration's image
        clear_output(wait=True)
        plt.imshow(image.astype(np.uint8), cmap='gray')
        plt.title(f"Iteration {it+1}")
        plt.axis('off')
        display(plt.gcf())
        
    plt.close()
    return image.astype(np.uint8)

# Main execution:

# 1. Generate clean synthetic image
clean_img = generate_clean_image((100, 100))

# 2. Add noise to create noisy image
noisy_img = add_noise(clean_img, noise_level=30)

# 3. Run PCA with visualization per iteration
denoised_img = parallel_cellular_algorithm_visual(noisy_img, iterations=50, mutation_rate=0.05)

# 4. Show final results side by side
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.title("Clean Image")
plt.imshow(clean_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Denoised Image (Final)")
plt.imshow(denoised_img, cmap='gray')
plt.axis('off')

plt.show()
