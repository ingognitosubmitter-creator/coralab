import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return np.exp(-0.85 * (x**2 + y**2))

# Define grid
x = np.linspace(-1, 1, 400)
y = np.linspace(-1, 1, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create the 2D color map
plt.figure(figsize=(6, 5))
img = plt.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')

# Add colorbar and labels
plt.colorbar(img, label=r"$f(x, y) = e^{-0.85\sqrt{x^2 + y^2}}$")
if False:
    plt.title("2D Color Map of $f(x, y)$")
    plt.xlabel("x")
    plt.ylabel("y")

plt.show()