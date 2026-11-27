from pathlib import Path
from src.load_documents import load_text_file,DATA_DIR

def chunk_tex_data(text, chunk_size=300, overlap=50):
    """
    split text into smaller text rather than passing whole text
    parameters:
            tetxt: input text
            chunk_size: maximum characters in chunk 
            overlap: number of shared characters between chunks
    return:
            list of text chunks
            
    """
    
    if overlap>=chunk_size:
        raise ValueError("Overlap must be smallerthan chunk size ")
    
    chunks=[]
    
    step_size = chunk_size - overlap
    for start in range(0, len(text),step_size):
        end = start+chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    file_path = DATA_DIR / "medical_ai.txt"
    document_text = load_text_file(file_path)
    
    chunks = chunk_tex_data(document_text,chunk_size=300, overlap=50)
    
    print(f"total chunk: {len(chunks)}")
    for i,chunk in enumerate(chunks):
        print(f"\n---chunk {i+1}---")
        print(chunk)
        print(f"characters:{len(chunks)}")
    
    
    