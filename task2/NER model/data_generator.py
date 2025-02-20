import json

#list of animals
animals = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

#data generator
def generate_sentences(animals):
    sentences = []

    #templates
    sentence_templates = [
        "There is a {} in the field.",
        "I saw a {} near the barn.",
        "A wild {} was spotted in the forest.",
        "Look at that {}!",
        "I have a {} as a pet.",
        "The {} is playing in the yard.",
        "Do you see the {} over there?",
        "A {} is running fast."
    ]

    for animal in animals:
        for template in sentence_templates:
            sentence_type = template.format(animal)
            tokens = sentence_type.split()
            ner_tags = [0] * len(tokens)
            for i, token in enumerate(tokens):
                if token.lower() == animal:
                    ner_tags[i] = 1
            sentences.append({"tokens": tokens, "ner_tags": ner_tags})

    return sentences

data = generate_sentences(animals)

#saving
with open('ner_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Dataset generated and saved in 'ner_dataset.json'")
