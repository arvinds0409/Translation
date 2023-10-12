import torch
from transformers import MarianTokenizer, MarianMTModel

# Load the pre-trained model and tokenizer for translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate 
def translate_to_hin(english_text):
    # Tokenize the input text
    input_ids = tokenizer.encode(english_text, add_special_tokens=True, return_tensors="pt")

    # Perform translation
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)

    # Decode the translated text
    hinglish_translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return hinglish_translation

while True:
    # Prompt the user to enter an English statement
    english_statement = input("Enter statement or type exit: ")

    if english_statement.lower() == 'exit':
        break

    # Translate the user's input to Hinglish
    hinglish_translation = translate_to_hin(english_statement)

    # Print the Hinglish translation
    print("Translated:", hinglish_translation)
