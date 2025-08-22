#%%
import pickle
import pandas as pd
import matplotlib.pyplot as plt

domain = ['Academia', 'Finance', 'Government', 'Law', 'News']
domain_n = [27.5, 26.1, 13.4, 17.3, 15.6]
qa_pair = ['Text-only', 'Multimodal', 'Meta data', 'Unanswerable']
qa_pair_n = [37.4, 27.9, 23.4, 11.3]
question = ['What/Who/Where/When/Which', 'Y/N', 'How', 'Why']
question_n = [58.6, 22.1, 18.8, 0.5]
answer = ['Numerical', 'Textual', 'Boolean', 'Others']
answer_n = [37.4, 35.7, 17.3, 9.6]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Dataset distribution based on different classification criteria", fontsize=16, fontweight='bold')

# Subplot 1: Domain
axs[0, 0].bar(domain, domain_n, color='skyblue')
axs[0, 0].set_title("Domain Distribution")
axs[0, 0].set_ylabel("Percentage (%)")

# Subplot 2: QA Pair Types
axs[0, 1].bar(qa_pair, qa_pair_n, color='lightgreen')
axs[0, 1].set_title("QA Pair Types")
axs[0, 1].set_ylabel("Percentage (%)")
axs[0, 1].tick_params(axis='x', rotation=15)

# Subplot 3: Question Types
axs[1, 0].bar(question, question_n, color='salmon')
axs[1, 0].set_title("Question Types")
axs[1, 0].set_ylabel("Percentage (%)")

# Subplot 4: Answer Types
axs[1, 1].bar(answer, answer_n, color='plum')
axs[1, 1].set_title("Answer Types")
axs[1, 1].set_ylabel("Percentage (%)")

# Tight layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("data.png")

# --------------------------------------------------------------------------------------------
#%%
import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open("training_loss.pkl", "rb") as f:
    training_loss = pickle.load(f)

clean_data = [entry for entry in training_loss if isinstance(entry, tuple)]

# Convert to DataFrame
df = pd.DataFrame(clean_data, columns=["round", "client_id", "loss"])

avg_loss = df.groupby("round")["loss"].mean()
print(avg_loss)

plt.figure(figsize=(10, 5))
plt.plot(avg_loss.index, avg_loss.values, marker='o')
plt.title("Average Training Loss per Round")
plt.xlabel("Round")
plt.ylabel("Average Loss")
plt.grid(True)
plt.savefig("loss_avg.png")

plt.figure(figsize=(12, 6))
for client_id in sorted(df["client_id"].unique()):
    print("client", client_id)
    client_data = df[df["client_id"] == client_id]
    if client_id == 0:
        plt.plot(client_data["round"], client_data["loss"], marker='o', label=f"Admin")
    plt.plot(client_data["round"], client_data["loss"], marker='o', label=f"Client {client_id}")

plt.title("Client-wise Training Loss over Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend(title="Client Name", loc="upper right")
plt.grid(True)
plt.savefig("loss_client.png")

# Count number of occurrences per client
client_counts = df["client_id"].value_counts().sort_index()

# Convert to DataFrame for easy viewing or LaTeX export
client_table = pd.DataFrame({
    "Client ID": client_counts.index,
    "Rounds Participated": client_counts.values
})

print(client_table)

latex_rows = client_table.to_latex(index=False)
print(latex_rows)

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

all_means = []

for user_idx in range(12):
    filename = f"BLEU_scores/scores_{user_idx+1}.csv"
    df = pd.read_csv(filename)

    if 'index' in df.columns:
        df = df.drop(columns=["index"])

    means = df.mean(numeric_only=True)  # get mean per column, numeric only

    means['user'] = f"user {user_idx}"  # add user id as a column
    all_means.append(means)

# Combine all means into one DataFrame
mean_df = pd.DataFrame(all_means)

# Set 'user' as the first column
cols = ['user'] + [col for col in mean_df.columns if col != 'user']
mean_df = mean_df[cols]

# Extract columns as lists
user_ids = mean_df['user'].tolist()
metric_columns = mean_df.columns.drop('user')

metrics_as_lists = {col: mean_df[col].tolist() for col in metric_columns}

print("Users:", user_ids)
for metric, values in metrics_as_lists.items():
    print(f"{metric}: {values}")

metrics = ["context_precision", "answer_relevancy", "faithfulness", "context_recall"]

all_means = []

for i in range(1, 13):
    df = pd.read_csv(f"RAGAS_scores/results_{i}.csv")

    mean_vals = [df[metric].mean() for metric in metrics]
    mean_dict = {metric: val for metric, val in zip(metrics, mean_vals)}
    mean_dict['user'] = f"user {i-1}"  # assuming user indexing starts at 0
    
    all_means.append(mean_dict)

mean_df_ra = pd.DataFrame(all_means)

# Extract lists for each metric and user IDs
user_ids = mean_df_ra['user'].tolist()
metrics_as_lists = {metric: mean_df_ra[metric].tolist() for metric in metrics}

print("Users:", user_ids)
for metric, values in metrics_as_lists.items():
    print(f"{metric}: {values}")

users = mean_df["user"]

data = {
    'BLEU': mean_df["BLEU"].tolist(),
    'ROUGE1-P': mean_df["ROUGE-ROUGE1-precision"].tolist(),
    'ROUGE1-R': mean_df["ROUGE-ROUGE1-recall"].tolist(),
    'ROUGE1-F': mean_df["ROUGE-ROUGE1-fmeasure"].tolist(),
    'ROUGEl-P': mean_df["ROUGE-ROUGEL-precision"].tolist(),
    'ROUGEl-R': mean_df["ROUGE-ROUGEL-recall"].tolist(),
    'ROUGEl-F': mean_df["ROUGE-ROUGEL-fmeasure"].tolist(),
    'Context_Precision': mean_df_ra["context_precision"].tolist(),
    'Answer_Relevancy': mean_df_ra["answer_relevancy"].tolist(),
    'Faithfulness': mean_df_ra["faithfulness"].tolist(),
    'Context_Recall': mean_df_ra["context_recall"].tolist()
}

print(data)

df = pd.DataFrame(data, index=users)

# Select metrics you want to plot (can be all or a subset)
metrics_to_plot = ['Context_Precision', 'Answer_Relevancy', 'Faithfulness', 'Context_Recall']

x = np.arange(len(users))
width = 0.12

fig, ax = plt.subplots(figsize=(15,7))

for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i*width, df[metric], width, label=metric)

ax.set_xticks(x + width * (len(metrics_to_plot)-1) / 2)
ax.set_xticklabels(users, rotation=45, ha='right')
ax.set_ylabel('Scores')
ax.set_title('User Metric Scores Comparison')
ax.legend()
plt.tight_layout()
plt.show()
