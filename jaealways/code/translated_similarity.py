from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import pandas as pd
from tqdm import tqdm


# Function to encode sentences
def encode_sentence(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].numpy()


if __name__ == '__main__':
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode sentences
    
    df=pd.read_csv('../data/train_translated.csv')
    list_sentence1 = df.iloc[:,1].tolist()
    list_sentence2 = df.iloc[:,2].tolist()

    list_similarity = []
    for s1, s2 in tqdm(zip(list_sentence1, list_sentence2)):
        if isinstance(s1, str) and isinstance(s2, str):
            embedding1 = encode_sentence(s1, tokenizer, model)[0]
            embedding2 = encode_sentence(s2, tokenizer, model)[0]

            # Calculate cosine similarity
            list_similarity.append(1 - cosine(embedding1, embedding2))
        else:
            list_similarity.append(None)  # Or handle non-string cases as you see fit


    df['similarity']=list_similarity
    df.to_csv('../data/train_translated_labeled.csv', index=False)

