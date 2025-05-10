from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os


class ChatBot():
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Load and split documents
        loader = TextLoader('./horoscope.txt', encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Create embeddings for the documents
        embeddings = HuggingFaceEmbeddings()

        # Use FAISS as the vector store (local, no cloud needed)
        docsearch = FAISS.from_documents(docs, embeddings)

        # Set up the language model from HuggingFace Hub
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define a clear and strict prompt for the LLM
        from langchain import PromptTemplate

        template = """
You are a horoscope assistant. Use ONLY the following context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Set up the RAG (Retrieval Augmented Generation) chain
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        self.rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
