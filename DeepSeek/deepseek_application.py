import torch
import logging
import docx
import PyPDF2
import chardet
import time
import os
import re
import pickle
import faiss
from langchain_community.vectorstores import FAISS   
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from utils.prompt_template import PromptHelper
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

@dataclass
class Metadata:
    """Metadata for processed documents"""
    filename: str
    chunk_count: int
    total_tokens: int
    processing_time: float

class Processor:
    def __init__(self):
        self.supported_extensions = {
            'pdf': self.process_pdf,
            'docx': self.process_docx,
            'txt': self.process_text,
            'md': self.process_text,
            'csv': self.process_text
        }

    def process_docx(self, document) -> str:
        try: 
            doc = docx.Document(document)
            content = []
            for section in doc.sections:
                header = section.header
                if header: 
                    content.append(paragraph.text for paragraph in header.paragraphs)
            
            content.append(paragraph.text for paragraph in doc.paragraphs)

            for table in doc.tables: 
                for row in table.rows: 
                    content.append(cell.text for cell in row.cells)
            
            return '\n'.join(filter(None, content))
        
        except Exception as e: 
            logger.error(f"Error processing DOCX: {str(e)}")
            raise ValueError(f"Failed to process DOCX: {str(e)}")

    def process_pdf(self, document) -> str:
        try: 
            pdf_reader = PyPDF2.PdfReader(document)
            return '\n'.join(
                page.extract_text().strip() for page in pdf_reader.pages if page.extract_text().strip()
            )        
        except Exception as e: 
            logger.error(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")

    def process_text(self, document):
        try: 
            data = document.getvalue()
            result = chardet.detect(data)
            encodings = [result['encoding'], 'utf-8', 'latin-1', 'ascii']
                
            for encoding in encodings:
                try:
                    if encoding:
                        return data.decode(encoding)
                except UnicodeDecodeError:
                    continue
                
                raise ValueError("Unable to decode file with any supported encoding")
        except Exception as e:
            logger.error(f"Error processing text file: {str(e)}")
            raise ValueError(f"Failed to process text file: {str(e)}")
        
    def process_file(self, document) -> Tuple[str, Metadata]:
        start_time = time.time()
        
        # Extract the file extension and name
        file_extension = os.path.splitext(document)[1].lower().lstrip('.')
        file_name = os.path.splitext(document)[0].split('/')[-1]

        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        processor = self.supported_extensions[file_extension]
        content = processor(document)
        
        # Calculate basic metrics
        metadata = Metadata(
            filename=file_name,
            chunk_count=len(content.split('\n')),
            total_tokens=len(content.split()),
            processing_time=time.time() - start_time
        )
        
        return content, metadata, file_name

class DeepSeekApplication:
    def __init__(
        self,
        client_id: int, 
        ori_model,
        lora_weights_path,
        lora_config_path,
        prompt_template,
        chunk_size: int = 1000, # Chunk size
        chunk_overlap: int = 0, # Chunk overlap
    ):
        self.client_id = client_id
        self.ori_model = ori_model
        self.lora_weights_path = lora_weights_path
        self.lora_config_path = lora_config_path
        self.prompt_template = prompt_template
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.init_model()
        self.doc_processor = Processor()
        self.document_store = None
        self.document_metadata = {}
        
        with open(self.lora_config_path + "/client_{}.pkl".format(self.client_id), "rb") as f:
            self.client = pickle.load(f)
        
        with open(self.lora_config_path + "/user_permission_resource.pkl", "rb") as f:
            user_permissions_resource = pickle.load(f)
        
        self.client.set_managers(user_permissions_resource)
    
    def init_model(self):
        self.prompter = PromptHelper(self.prompt_template)
        config = AutoConfig.from_pretrained(self.ori_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.ori_model,
            config=config,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
        )

        self.recursive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
        )

        model_kwargs = {
            'device': device
        }
        # Ensure that there are no out of memory issues
        encode_kwargs = { 
            'normalize_embeddings': True, 
            'batch_size': 8
        }
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", # Embedding is different from the original model due to efficient embedding usage
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Loading the model
        self.model = prepare_model_for_kbit_training(self.model)
        lora_weights = torch.load(self.lora_weights_path) # Get the LoRA weights
        config_peft = LoraConfig.from_pretrained(self.lora_config_path)
        self.model = PeftModel(self.model, config_peft)
        set_peft_model_state_dict(self.model, lora_weights, "default")

        self.tokenizer = AutoTokenizer.from_pretrained(self.ori_model)    
        self.tokenizer.pad_token_id = (
            0
        )
        self.tokenizer.padding_side = "left"

        self.model.half()
        self.model.eval() # Set the model to evaluation mode 
    
    def retrieve_relevant_docs(self, question: str, top_k: int, sim_threshold: float) -> List[str]:
        if self.document_store is None: 
            raise ValueError("There are no documents uploaded.")
        
        try: 
            scores = self.document_store.similarity_search(
                query=question, 
                k=top_k
            )

            text_splits = self.recursive_text_splitter.split_documents([scores[0]])
            vectorstore = Chroma.from_documents(documents=text_splits, embedding=self.embeddings)

            self.retriever = vectorstore.as_retriever()

            return scores[0].page_content

        except Exception as e: 
            logger.error(f"Error retrieving relevant documents: {str(e)}")
            raise
    
    def preprocess_file(self, document: str) -> str: 
        document = ' '.join(document.split())
        document = document.replace('\t', ' ').strip()
        return document
    
    def load_documents(self, documents: List[str], metadata: Optional[Dict[str, Metadata]] = None) -> None:
        try:
            doc_chunks = []
            self.documents = documents
            documents_array = []

            def loading_documents(documents: List, documents_array: List, dict: bool = False):
                # Getting the documents content into the Document Langchain object
                if dict: 
                    documents_array = (
                        Document(
                            page_content=doc["context"], 
                            metadata={"space_key_index": doc["space_key_index"]}
                        )
                        for doc in documents
                    )
                else: 
                    documents_array = (
                        Document(
                            page_content=doc, 
                            metadata=metadata
                        )
                        for doc in documents
                    )
                return documents_array
            
            if documents: 
                self.documents_array = loading_documents(documents, documents_array) # Adding additional documents to the chunks
            self.documents_array = loading_documents(self.client.get_documents(), documents_array, dict=True) # Adding the documents of the clients they have access to

            splitted_docs = self.text_splitter.split_documents(self.documents_array)
            self.document_store = FAISS.from_documents(splitted_docs, self.embeddings)
            
            if metadata:
                self.document_metadata.update(metadata)
            
            logger.info(f"Successfully loaded {len(doc_chunks)} chunks from {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise    

    def generate_response(
        self,
        query: str, # The question to be asked
        deepseek, # DeepSeekApplication instance for processing documents
        top_k: int, # Number of highest probability vocabulary tokens to keep for top-k filtering
        top_p: int, # The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
        num_beams: int, # Number of beams for beam search
        max_new_tokens: int,
        similarity_threshold: float,
        temp: float,
        context: bool, # Boolean to check if there are documents for context
        max_context_length: int = 2000 # The maximum tokens for the input context 
    ):
        start_time = time.time()
        
        try:
            context_documents = self.retrieve_relevant_docs(query, top_k, similarity_threshold)

            retrieved_bits = self.retriever.get_relevant_documents(query)
            texts = [
                doc.page_content for doc in retrieved_bits
            ]

            combined_texts = ' '.join(texts)

            from transformers import pipeline
            from langchain_huggingface.llms import HuggingFacePipeline
            from langchain.chains import ConversationalRetrievalChain
            from langchain_core.prompts import ChatPromptTemplate

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )


            pipe = pipeline("question-answering", model=deepseek.model, tokenizer=deepseek.tokenizer, return_full_text=True)
            llm = HuggingFacePipeline(pipeline=pipe)

            preds = llm(
                question=query, 
                context=retrieved_bits,
            )
            print(
                f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
            )

            # qa_chain = ConversationalRetrievalChain.from_llm(
            #     llm=llm,
            #     retriever=self.retriever, 
            #     condense_question_prompt=ChatPromptTemplate.from_messages([
            #         ("system", system_prompt),
            #         ("human", "{question}"),
            #     ]),
            #     return_source_documents=True,
            # )

            # results = qa_chain.invoke({
            #     "question": query, 
            #     "chat_history": [],
            #     })
            
            print("other results", results)

            
            # Truncate context if it is too long
            if len(combined_texts) > max_context_length:
                combined_context = combined_texts[:max_context_length] + "..."
            
            prompt = self.construct_prompt(query, combined_context)
            print("prompt", prompt)
            inputs = deepseek.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            generation_config = GenerationConfig(
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams
            )
            with torch.no_grad():
                generated_output = deepseek.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            s = generated_output.sequences[0]
            output = deepseek.tokenizer.decode(s)

            answer = {
                'content': output,
                'metadata': {
                    'processing_time': time.time() - start_time,
                    'context_length': len(combined_context),
                    'query_length': len(query)
                }
            }
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def post_processing(self, output: str) -> str:
        answer = re.split(r"Answer:*?", output) # Extracting the answer
        think_answer = re.split(r"</think>", answer[2]) # Removing the think caps
        final_answer = re.split(r"<｜end▁of▁sentence｜>", think_answer[1]) # Removing the end of sentence token
        
        return final_answer[0].split('\n')[-1]

    def construct_prompt(self, query: str, context: str) -> str:
        """Construct an enhanced prompt template"""
        return f"""
        You are given a context document and a related question. Your task is to generate a comprehensive answer based on the context.

        Context:
        {context}

        Question:
        {query}

        Instructions:
        - Answer based only on the given context if it's relevant.
        - If the context is insufficient or empty, provide the best answer using your own knowledge.
        - Based on the context above, explain your answer in complete sentences.
        - Ensure your answer is:
        1. Directly relevant
        2. Accurate and fact-based
        3. Complete and informative
        4. Clear and well-structured

        Please provide a full-sentence answer.
        """