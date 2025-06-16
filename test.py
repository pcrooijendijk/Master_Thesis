import numpy as np
import os

output_dir = "FL_output"

training_loss = np.load(os.path.join(output_dir, "training_loss.npy"))

import matplotlib.pyplot as plt
import numpy as np

losses = np.load("path/to/training_loss.npy")

plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss.png")
