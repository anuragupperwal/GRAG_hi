import os
import fasttext
import fasttext.util

def expand_query_with_fasttext(query, model_path="cc.hi.300.bin", top_n=5):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FastText model not found at {model_path}")
    
    ft_model = fasttext.load_model(model_path)
    words = query.strip().split()
    expanded_words = set(words)
    
    for word in words:
        try:
            neighbors = ft_model.get_nearest_neighbors(word)
            expanded_words.update([w for _, w in neighbors[:top_n]])
        except Exception as e:
            print(f"Error expanding word '{word}': {e}")
    
    return " ".join(expanded_words)

