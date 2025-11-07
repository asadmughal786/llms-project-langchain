import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from speech_to_text import speech_to_text

load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.4)

def read_doc(directory):
    loader = PyPDFDirectoryLoader(directory)
    return loader.load()

def chunk_data(docs, chunk_size=800, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def process_in_batches(documents, batch_size=50):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def main():
    print("Reading documents...")
    docs = read_doc("Documents/")
    print("Docs:", len(docs))

    print("Chunking...")
    chunks = chunk_data(docs)
    print("Chunks:", len(chunks))

    print("Embeddings...")
    embeddings = OpenAIEmbeddings()

    print("Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "llms-project"

    existing = [i["name"] for i in pc.list_indexes()]
    print("Indexes:", existing)

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created!")

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Upload only if empty
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) == 0:
        print("Uploading vectors...")
        for batch in process_in_batches(chunks, 50):
            vector_store.add_documents(batch)
            print(f"Added batch {len(batch)}")

    print("Building retriever & QA chain...")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    while True:
        question = input("\nEnter your query (Q to quit): ")
        if question.lower() == "q":
            break

        print("Searching...")
        answer = qa_chain.invoke({"query": question})
        text = answer["result"]
        print("Answer:", text)

        try:
            speech_to_text(text=text)
        except:
            pass


if __name__ == "__main__":
    main()
