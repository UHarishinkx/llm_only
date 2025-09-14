
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

def generate_variations(query_content):
    """Generates variations of a query."""
    variations = [
        query_content,
        query_content.replace("Find", "Show me"),
        query_content.replace("all", ""),
        query_content.replace("ARGO floats", "floats"),
        "Can you " + query_content.lower(),
        query_content + " please?",
        query_content.replace("within", "inside"),
        query_content.replace("radius", "area"),
        query_content.replace("°N", " degrees North"),
        query_content.replace("°E", " degrees East"),
        "Give me the data for " + query_content.lower(),
        query_content.upper(),
        query_content.replace("Show me", "Display"),
        query_content.replace("full", "complete"),
        query_content.replace("trajectory", "path"),
        "Plot the " + query_content.lower(),
        query_content.replace("measurements", "data"),
        query_content.replace("within", "in"),
        "What is the " + query_content.lower(),
        "Tell me about " + query_content.lower(),
    ]
    return list(set(variations)) # Return unique variations

def test_similarity():
    """Tests the semantic similarity of prompts."""
    try:
        with open("C:\\Users\\USER\\Desktop\\dead\\copy2\\cleaning\\argo_expanded_vectordb_iteration\\semantic_samples\\geographic_spatial.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: geographic_spatial.json not found.")
        return

    queries = data.get("queries", [])
    if not queries:
        print("No queries found in the JSON file.")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for query in queries:
        query_id = query.get("id")
        content = query.get("content")
        if not content:
            continue

        variations = generate_variations(content)
        if len(variations) < 2:
            print(f"Could not generate enough variations for query: {query_id}")
            continue

        embeddings = model.encode(variations, convert_to_tensor=True)
        
        # Calculate cosine similarity between the first (original) embedding and all others
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
        
        # Convert to a list of floats
        scores = cosine_scores.cpu().numpy().flatten().tolist()
        
        avg_score = np.mean(scores)

        print(f"--- Query ID: {query_id} ---")
        print(f"Original Content: {content}")
        print(f"Number of Variations Tested: {len(variations) - 1}")
        print(f"Average Similarity Score: {avg_score:.4f}")
        if avg_score < 0.85:
            print("  -> Score is below 0.85. Consider refining the prompt.")
        print("\n")

if __name__ == "__main__":
    test_similarity()
