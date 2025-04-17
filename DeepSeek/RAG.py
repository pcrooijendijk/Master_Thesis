from langchain_community.document_loaders import TextLoader
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

docs = []
loader = TextLoader("doc1.txt")
docs.extend(loader.load())
loader = TextLoader("doc2.txt")
docs.extend(loader.load())
loader = TextLoader("doc3.txt")
docs.extend(loader.load())
loader = TextLoader("doc4.txt")
docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

text_files = {
    "doc1.txt": ["E"],
    "doc2.txt": ["E", "S"],
    "doc3.txt": ["S"],
    "doc4.txt": ["E"],
}

for doc in documents:
    doc.metadata["role"] = text_files[doc.metadata["source"]]

embedding_model = OllamaEmbeddings(model="phi3:mini")
embedding_vectors = []

for doc in documents:
    embedding = embedding_model.embed_query(doc.page_content)
    role_value = 0
    if "E" in doc.metadata["role"]:
        role_value += 1
    if "S" in doc.metadata["role"]:
        role_value += 2
    extended_vector = np.concatenate([embedding, [role_value]])
    embedding_vectors.append(extended_vector)

embedding_array = np.array(embedding_vectors)

embedding_dim = len(embedding_vectors[0]) - 1
index = faiss.IndexFlatL2(embedding_dim + 1)
index.add(embedding_array)

class MetadataFAISSRetriever:
    def __init__(self, index, embedding_model, documents):
        self.index = index
        self.embedding_model = embedding_model
        self.documents = documents
    
    def retrieve(self, query, user_role):
        query_embedding = self.embedding_model.embed_query(query)
        
        if user_role == "E":
            query_role_value = int(format(1, '016b'), 2)
        elif user_role == "S":
            query_role_value = int(format(2, '016b'), 2)
        else:
            query_role_value = int(format(3, '016b'), 2)
        
        query_vector = np.concatenate([query_embedding, [query_role_value]])
        distances, indices = self.index.search(query_vector.reshape(1, embedding_dim + 1), k=5)
        
        retrieved_docs = []
        for i in indices[0]:
            if i < len(self.documents):
                doc_role_value = 0
                if "E" in self.documents[i].metadata["role"]:
                    doc_role_value += int(format(1, '016b'), 2)
                if "S" in self.documents[i].metadata["role"]:
                    doc_role_value += int(format(2, '016b'), 2)
                if query_role_value & doc_role_value:
                    retrieved_docs.append(self.documents[i])
        
        return retrieved_docs

    def score_documents(self, query, retrieved_docs):
        query_embedding = np.array(self.embedding_model.embed_query(query)).reshape(1, -1)
        
        doc_embeddings = []
        for doc in retrieved_docs:
            doc_embedding = np.array(self.embedding_model.embed_query(doc.page_content))
            doc_embeddings.append(doc_embedding)
        doc_embeddings = np.array(doc_embeddings)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        scored_docs = [(doc, similarity) for doc, similarity in zip(retrieved_docs, similarities)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        
        top_docs = [doc for doc, _ in scored_docs[:10]]
        return top_docs

retriever = MetadataFAISSRetriever(index, embedding_model, documents)

user_role = 'E'  # Current user's role
query = "What is the core architecture of our solution?"

if user_role == 'S':
    role_str = "Role: ['S'], so you are in Sales."
elif user_role == 'E':
    role_str = "Role: ['E'], so you are in Engineering."

llm = Ollama(model="phi3:mini", temperature=0.0)
promptList = []
promptPart1 = """
You are a system designed to provide information based on documents available to either the engineering team, the sales team, or both. Your task is to answer the user's question using only the context provided from the documents.
"""
promptPart2 = """
Ensure that:
1. If the relevant document(s) are accessible to the user's role, provide only the information directly from the document(s) that answers the query. Do not include any additional context or details that are not present in the document(s).
2. If the relevant document(s) are not accessible to the user's role, strictly state: 'Sorry, I can't share this information as you do not have access.'
3. Avoid adding any extra details, speculative information, prior content, or context beyond what is directly contained in the document(s).
4. If the document is accessible by multiple roles, validate access accordingly, but do not infer or combine roles.

**Important**
- Provide a response that is solely based on the document content relevant to the query.
- Exclude any information that is not present in the document(s) provided.
<context>
{context}
</context>
Question: {input}
"""
promptList.append (promptPart1)
promptList.append (role_str)
promptList.append (promptPart2)
finalPrompt = "".join (promptList)
prompt = ChatPromptTemplate.from_template (finalPrompt)

from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

retrieved_docs = retriever.retrieve(query, user_role)
most_relevant_docs = retriever.score_documents(query, retrieved_docs)

response = document_chain.invoke({"input": query, "context": most_relevant_docs})

print(response)