from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentence = "A dog is playing in the garden"

embedding = model.encode(sentence)
print("embeddings:")
print(embedding)

print(f"embedding type: {type(embedding)}")
