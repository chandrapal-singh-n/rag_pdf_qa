import numpy as np
from sentence_transformers import SentenceTransformer

# load embedding model
model =SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



# Step 1: Load the embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Prepare our document sentences
documents = [
    "CNN is a deep learning architecture used for image processing.",
    "A dog is playing in the garden.",
    "U-Net is commonly used for medical image segmentation.",
    "Transformers are widely used in natural language processing.",
    "Medical imaging helps doctors diagnose diseases."
]

# Step 3: Convert documents into embeddings
document_embeddings = model.encode(documents)

# Step 4: Get the user's question
# query = "Which architecture is used for medical image segmentation?"
# query = "Where does the animal run?"
# query = "How can tumor be cured?"
query = "What are autonoumous vehicles?"

# Step 5: Convert the question into an embedding
query_embedding = model.encode(query)

# Step 6: Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Step 7: Compare the question with every document
scores = []

for i in range(len(documents)):
    score = cosine_similarity(
        query_embedding,
        document_embeddings[i]
    )

    scores.append(score)

# Step 8: Rank documents by similarity
ranked_indices = np.argsort(scores)[::-1]

# Step 9: Display the results
print("\nUser question:", query)
print("\nMost relevant documents:\n")

for index in ranked_indices:
    print("Similarity:", round(scores[index], 4))
    print("Document:", documents[index])
    print("-" * 60)
