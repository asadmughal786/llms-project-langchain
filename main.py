import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import openai
from speech_to_text import speech_to_text

# Load environment variables from .env
load_dotenv()

# llms model making
llm = OpenAI(model_name ='gpt-3.5-turbo-instruct', temperature=0.5)
chain = load_qa_chain(llm,chain_type='stuff')

# Reading the document files
def read_doc(directory):
    """
    Reads all PDF files from a directory and returns the documents.
    """
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Converting the document into text chunks
def chunk_data(docs, chunk_size=500, chunk_overlap=100):
    """
    Splits documents into smaller chunks to avoid token limits.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def similarity_search():
    pass

# cosine similarity retrieve results from VectorDB (Pinecone)
def retrieve_query(query, index, k=2):
    try:
        matching_results = index.similarity_search(query, k=k)
        return matching_results
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []

# Search answers from VectorDB
def retrieve_answers(query, index):
    doc_search = retrieve_query(query, index)
    if not doc_search:
        return "No matching documents found."
    response = chain.run(input_documents=doc_search, question=query)
    return response

# Batch documents for processing
def process_in_batches(documents, batch_size=10):
    """
    Processes documents in smaller batches to reduce rate-limiting issues.
    """
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

# Main script
def main():
    # Step 1: Read and split documents
    print("Reading documents...")
    doc = read_doc("Documents/")
    print('Number of documents:', len(doc))

    print("Chunking documents...")
    documents = chunk_data(docs=doc)
    print(f"Number of chunks created: {len(documents)}")

    # for i, doc in enumerate(documents[:5]):
    #     print(f"Chunk {i}: {doc.page_content[:200]}...")

    # Step 2: Initialize OpenAI embeddings
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))

    # Step 3: Initialize Pinecone
    print("Initializing Pinecone...")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "llms-project"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    print('Existing indexes:', existing_indexes)

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index {index_name} created!")
        index = pc.Index(index_name)

        # Step 4: Add documents to Pinecone in batches
        print("Adding documents to Pinecone...")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)

        for batch in process_in_batches(documents, batch_size=10):
            try:
                vector_store.add_documents(documents=batch)
                print(f"Successfully added a batch of {len(batch)} documents.")
                time.sleep(1)  # Add delay to avoid rate limits
            except openai.error.RateLimitError as e:
                print("Rate limit exceeded. Retrying in 5 seconds...")
                time.sleep(5)  # Retry after a delay
            except Exception as e:
                print(f"Error adding documents: {e}")

        print("Documents successfully added to Pinecone!")
    else:
        print(f"Index {index_name} already exists!")

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    # Step 2: Retrieve answers from the index
    while True:
        query = input("Enter your query: ")
        print('Do you want to quit this app? press "Q" to quit the application')
        if query == 'q' or query=='Q':
            return False
        print("Searching for relevant documents...")
        answer = retrieve_answers(query, vector_store)
        # print(type(answer))
        speech_to_text(text=answer)
        # print("Answer:", answer)

if __name__ == "__main__":
    main()
