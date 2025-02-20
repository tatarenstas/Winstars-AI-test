import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load model
def load_ner_model(model_path, tokenizer_path):
    model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    tokenizer = torch.load(tokenizer_path)
    
    return model, tokenizer

def extract_animals(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens).logits
    predictions = torch.argmax(outputs, dim=2)
    print(predictions)

    tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    print(tokenized_text)
    
    animals = [tokenized_text[i] for i, label in enumerate(predictions[0]) if label == 1]
    return animals

if __name__ == "__main__":
    model_path = "ner_model.pth"
    tokenizer_path = "ner_tokenizer.pth"
    
    model, tokenizer = load_ner_model(model_path, tokenizer_path)
    
    text = "There is a cow in the picture."
    animals = extract_animals(text, model, tokenizer)
    
    if animals:
        print("Animals found:", animals)
    else:
        print("No animals found.")
