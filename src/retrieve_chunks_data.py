import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
from src.similarity_calc import cosine_similarity

def retrieve_chunks(query, chunks, chunk_embeddings, top_k=3):

    # Convert user question into embedding
    query_embedding = model.encode(query)

    # Store similarity scores
    scores = []

    # Compare query with every chunk
    for i in range(len(chunks)):

        score = cosine_similarity(
            query_embedding,
            chunk_embeddings[i]
        )

        scores.append(score)

    # Rank chunks from highest to lowest score
    ranked_indices = np.argsort(scores)[::-1]

    # Select top K chunks
    top_indices = ranked_indices[:top_k]

    # Collect retrieved chunks
    results = []

    for index in top_indices:

        result = {
            "chunk": chunks[index],
            "score": scores[index]
        }

        results.append(result)

    return results