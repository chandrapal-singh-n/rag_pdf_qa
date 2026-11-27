import numpy as np
from sentence_transformers import SentenceTransformer

from src.load_documents import load_text_file, DATA_DIR
from src.chunking import chunk_tex_data
from src.similarity_calc import cosine_similarity
from src.retrieve_chunks_data import retrieve_chunks
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


file_path = DATA_DIR / "review_article_prompt.txt"

document_text = load_text_file(file_path)

chunks = chunk_tex_data(document_text, chunk_size=300, overlap=20)

chunk_embedding = model.encode(chunks)

print(f"Embedding shape {chunk_embedding.shape}")



if __name__ == "__main__":

    query = "What is Structure?"

    results = retrieve_chunks(
        query,
        chunks,
        chunk_embedding,
        top_k=3
    )

    print("\nUser question:", query)
    print("\nRetrieved results:\n")

    for i, result in enumerate(results, start=1):

        print(f"Result {i}")
        print("Similarity:", round(result["score"], 4))
        print("Chunk:", result["chunk"])
        print("-" * 60)


