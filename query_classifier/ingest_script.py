import json, os

def get_vector(text: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5]

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
    filename1 = "data_files/rag_examples.json"
    filename2 = "data_files/profile_examples.json"
    
    try:
        generate_embeddings(filename1)
        generate_embeddings(filename2)
    except Exception as e:
        print(f"Error in main: {e}")
    
if __name__ == "__main__":
    main()