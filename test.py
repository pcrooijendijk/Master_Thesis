import numpy as np
import os

training_loss = np.load("training_loss.npy")

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 4))
plt.plot(training_loss, label='Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss.png")

print(training_loss)