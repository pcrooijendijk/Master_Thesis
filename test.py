import numpy as np
import pickle

with open("training_loss3.pkl", "rb") as f:
    training_loss = pickle.load(f)

print(training_loss)

# import matplotlib.pyplot as plt
# import numpy as np

# # Plot each sublist
# plt.figure(figsize=(10, 6))
# for i, sublist in enumerate(training_loss):
#     plt.plot(sublist, label=f"Series {i}")

# plt.xlabel("Index within series")
# plt.ylabel("Loss value")
# plt.title("Loss Values over Time for Each Series")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig("loss.png")

import matplotlib.pyplot as plt

rounds = sorted(training_loss.keys())
avg_losses = [sum(training_loss[r]) / len(training_loss[r]) for r in rounds]

plt.plot(rounds, avg_losses, marker='o')
plt.xlabel("Communication Round")
plt.ylabel("Average Training Loss")
plt.title("Average Training Loss per Round")
plt.grid(True)
plt.show()
