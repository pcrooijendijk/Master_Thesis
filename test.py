import numpy as np
import pickle

with open("training_loss.pkl", "rb") as f:
    training_loss = pickle.load(f)

import matplotlib.pyplot as plt
import numpy as np

# Plot each sublist
plt.figure(figsize=(10, 6))
for i, sublist in enumerate(training_loss):
    plt.plot(sublist, label=f"Series {i}")

plt.xlabel("Index within series")
plt.ylabel("Loss value")
plt.title("Loss Values over Time for Each Series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("loss.png")