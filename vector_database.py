from sentence_transformers import SentenceTransformer
from data_processing import load_data_parallel
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore

texts, labels, speakers = load_data_parallel()

# Load a pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the documents
document_embeddings = embedding_model.encode(texts)

print(f"Generated embeddings shape: {document_embeddings.shape}")

# Create a FAISS index
embedding_dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dimension)

# Add embeddings to the index
faiss_index.add(np.array(document_embeddings))

print(f"FAISS index contains {faiss_index.ntotal} vectors.")

vector_store = FAISS(
    faiss_index=faiss_index,
    embedding_function=lambda x: embedding_model.encode([x])[0],
    docstore=InMemoryDocstore.from_texts(texts)
)

# query based on the fact the milk lobby was a big deal in this year in the congressional record
query = "What is better - full fat or skim milk?"

# Convert query to embedding
query_embedding = embedding_model.encode([query])

# Search for the most similar document
distances, indices = faiss_index.search(np.array(query_embedding), k=1)

# Display the result
result = texts[indices[0][0]]
print(f"Query: {query}")
print(f"Retrieved Document: {result}")