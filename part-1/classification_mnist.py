from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml(name="mnist_784", as_frame=False)

X, y = mnist.data, mnist.target

print(f"Data: {X}, Targets: {y}")
print(f"Shapes: {X.shape}, {y.shape}") 
# returns (70000, 784) -> 70k images 28x28 where each pixel is a feature from 0 to 255 (black intensity)
# (70000) -> class of the image (they are handwritten numbers)

# peek of one of the images

def plot_digit(image_data):
    image = image_data.reshape(28, 28) # reshape as matrix
    plt.imshow(image, cmap="binary")
    plt.axis("off")

plot_digit(X[0])
plt.savefig("digit.png")