import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open("training_loss4.pkl", "rb") as f:
    training_loss = pickle.load(f)

clean_data = [entry for entry in training_loss if isinstance(entry, tuple)]

# Convert to DataFrame
df = pd.DataFrame(clean_data, columns=["round", "client_id", "loss"])

avg_loss = df.groupby("round")["loss"].mean()

plt.figure(figsize=(10, 5))
plt.plot(avg_loss.index, avg_loss.values, marker='o')
plt.title("Average Training Loss per Round")
plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig("loss_avg.png")

plt.figure(figsize=(12, 6))
for client_id in df["client_id"].unique():
    client_data = df[df["client_id"] == client_id]
    plt.plot(client_data["round"], client_data["loss"], marker='o', label=f"Client {client_id}")

plt.title("Client-wise Training Loss over Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_client.png")