from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
def tokenize_text(text):
    return tokenizer.tokenize(text)

# mapping tokens to their IDs
def text_to_ids(text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

def ids_to_text(ids):
    return tokenizer.convert_ids_to_tokens(ids)

def text_to_string(ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))

if __name__ == "__main__":
    # Example usage
    input_text = "Hello, how are you?"
    tokened_text = tokenize_text(input_text)
    text2ids = text_to_ids(input_text)
    ids2text = ids_to_text(text2ids)
    text2string = text_to_string(text2ids)
    print("Tokenized Text:", tokened_text)
    print("Token IDs:", text2ids)
    print("Text from IDs:", ids2text)
    print("String from IDs:", text2string)