from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")

amnesty_subset = amnesty_qa["eval"].select(range(2))

eval_dataset = amnesty_qa["eval"].select(range(1,3))
eval_dataset.to_pandas()

amnesty_subset.to_pandas()

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from langchain_community.chat_models import ChatOllama
from ragas import evaluate, RunConfig
from langchain_community.embeddings import OllamaEmbeddings
# information found here: https://docs.ragas.io/en/latest/howtos/customisations/bring-your-own-llm-or-embs.html

llm_llama3 = ChatOllama(model="tinyllama",verbose=False,timeout=600,num_ctx=8192,disable_streaming=False)
embeddings_llama3 = OllamaEmbeddings(model="tinyllama")

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=llm_llama3,
    embeddings=embeddings_llama3,
    run_config =RunConfig(timeout=600, max_retries=20, max_wait=50,log_tenacity=False),
    raise_exceptions=True
)

result.to_pandas()