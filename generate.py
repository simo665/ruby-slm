import torch
from model import RubyMini
from utils import tokenization  
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenization.tokenizer.vocab_size 
model = RubyMini(vocab_size).to(device)
model.load_state_dict(torch.load("weight_data/ruby_mini4_epoch_1.pth", map_location=device))
model.eval()

def top_k_logits(logits, k):
    """Apply top-k filtering to logits."""
    if k == 0:
        # no truncation
        return logits
    
    k = min(k, logits.size(-1))  # safety check
    values, _ = torch.topk(logits, k=k)
    min_values = values[..., -1, None]  # get the k-th largest value
    
    return torch.where(
        logits < min_values,
        torch.full_like(logits, -1e10),
        logits
    )

def top_p_logits(logits, p):
    """Apply nucleus (top-p) sampling to logits."""
    if p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability exceeds p
    # We want to keep tokens until cumsum > p, so we shift by 1
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep the first token that makes cumsum > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Create a mask for the original indices
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    
    return torch.where(indices_to_remove, torch.full_like(logits, -1e10), logits)

def generate_text(model, start_tokens, max_length=100, top_k=50, top_p=1.0, temperature=1.0):
    """Generate text using the model with top-k and top-p sampling."""
    generated = start_tokens.copy()  # list of token IDs
    
    for _ in range(max_length):
        # Use last 128 tokens as context (or whatever your model's max context is)
        input_ids = torch.tensor([generated[-512:]], dtype=torch.long).to(device)
        
        # Check if input is empty
        if input_ids.size(1) == 0:
            print("Error: Empty input sequence")
            return generated
            
        with torch.no_grad():
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            
        # Check if logits are empty
        if logits.size(1) == 0:
            print("Error: Model produced empty logits")
            return generated
        
        # Get logits for the last token and apply temperature
        last_token_logits = logits[0, -1] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            last_token_logits = top_k_logits(last_token_logits, top_k)
        
        # Apply top-p filtering  
        if top_p < 1.0:
            last_token_logits = top_p_logits(last_token_logits, top_p)
        
        # Sample from the filtered distribution
        probabilities = F.softmax(last_token_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1).item()

        # Stop if EOS token is generated
        if next_token_id == tokenization.tokenizer.eos_token_id:
            break

        
        generated.append(next_token_id)
        
        # Optional: stop if end token
        # if next_token_id == tokenizer.eos_token_id:
        #     break
    
    return generated

# Example usage
if __name__ == "__main__":
    
    while True:
        start_text = input("You: ")
        if start_text.lower() == 'exit':
            break
            
        # Get sampling parameters
        
        start_tokens = tokenization.text_to_ids(start_text)
        
        # Check if we have valid tokens
        if not start_tokens:
            print("Error: Could not tokenize input text")
            continue
            
        generated_ids = generate_text(
            model, 
            start_tokens, 
            max_length=30,
            temperature=1.0,
            top_k=50,
            top_p=1.0
        )
        
        # Convert generated ids back to text
        generated_text = tokenization.text_to_string(generated_ids)
        
        print("\nComputer:", generated_text)
        print("-" * 50)