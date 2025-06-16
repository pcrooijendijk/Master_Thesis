import numpy as np
import os

output_dir = "FL_output"

training_loss = np.load(os.path.join(output_dir, "training_loss.npy"))

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
