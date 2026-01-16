import json
from sentence_transformers import SentenceTransformer

# =====================
# Model (MUST match classifier)
# =====================
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def get_vector(text: str) -> list[float]:
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def generate_embeddings(filename: str):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            
        for item in data:
            text = item["text"]
            vector = get_vector(text)
            item["vector"] = vector
            
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
                
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} does not exist")
    except Exception as e:
        raise Exception(f"Error generating embeddings for {filename}: {e}")

def main():
    filename = "/Users/anhadhsran/Documents/homeify-ai-2.0/Collector-AI/data_files/examples.json"
    
    try:
        generate_embeddings(filename)
    except Exception as e:
        print(f"Error in main: {e}")
    
if __name__ == "__main__":
    main()