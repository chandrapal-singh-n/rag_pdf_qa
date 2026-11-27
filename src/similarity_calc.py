import numpy as np

def cosine_similarity(vector_a,vector_b):
    
    dot_prod = np.dot(vector_a,vector_b)
    
    magnitude_a= np.linalg.norm(vector_a)
    magnitude_b= np.linalg.norm(vector_b)
    
    if magnitude_a==0 or magnitude_b==0:
        return 0.0
    
    # calculate cosine 
    similarity_out = dot_prod/ (magnitude_a*magnitude_b)
    return similarity_out


if __name__ =="__main__":
    A= np.array([1,0])
    B=np.array([0.9,0.1])
    C=np.array([0,1])
    print("Similarity beween a and b:", 
          cosine_similarity(A,B))