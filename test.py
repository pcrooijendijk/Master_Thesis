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