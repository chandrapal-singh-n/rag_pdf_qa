from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"


def load_text_file(file_path):
    """
    read text file and return  its content
    """
    with open(file_path, "r",encoding="utf-8") as f:
        text= f.read()
    return text

if __name__=="__main__":
    file_path = DATA_DIR / "medical_ai.txt"
    
    try: 
        if not file_path.exists():
            raise FileNotFoundError('check file name or path ')
        document_text = load_text_file(file_path)
    except FileNotFoundError as e:
        print(f"error {e}")
        