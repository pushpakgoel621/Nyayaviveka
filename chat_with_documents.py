import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from utils import MEMORY, load_document

LLM = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature= 0.1,
    repetition_penalty = 1.03,
    max_new_tokens = 1024,
    top_k = 30,
    huggingfacehub_api_token = "change with your HF token",
    
)


def configure_retriever(
        docs
):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # alternatively: 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        },
    )
    return retriever


def configure_chain(retriever: BaseRetriever):
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000,
    )
    return ConversationalRetrievalChain.from_llm(
        **params
    )


def configure_retrieval_chain(
        uploaded_files,
):
    docs = []
    temp_dir = "temp_dir"
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs)
    chain = configure_chain(retriever=retriever)
    return chain
