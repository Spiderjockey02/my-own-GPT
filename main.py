import torch
from helpers.GPT import GPTLanguageModel
from helpers.Tokenizer import TokenProcessor
from torch.utils.data import DataLoader, TensorDataset

# Initialize TokenProcessor and GPT model
print("Loading Tokenizer and processing tokens...")
processor = TokenProcessor('./data/text.txt')
processor._load_text()
processor._process_text()

with open('./data/text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize GPT model with the tokenizer's vocabulary size
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.tensor(processor.encode(text), dtype=torch.long).to(device)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n].to(device)
val_data = data[n:].to(device)

model = GPTLanguageModel(processor.vocab_size, train_data, val_data)

# Command interface for interacting with the model
while True:
    text = input("Command: ")
    args = text.split(" ")
    cmd = args[0].lower()

    if cmd == "token":
        processor._load_text()
        processor._process_text()
        print("Text tokenized successfully.")
    elif cmd == "save":
        model.save_model("./data/model.pth")
        print("Model saved to './data/model.pth'.")
    elif cmd == "load":
        if len(args) < 2:
            print("Usaggenee: load <file_path>")
        else:
            model.load_model(args[1])
            print(f"Model loaded from '{args[1]}'.")
    elif cmd == "train":
        print("Starting training...")
        model.train_model()
        print("Training completed.")
    elif cmd == "eval":
        print("Evaluating model...")
        
        model.evaluate()
    elif cmd == "generate":
        if len(args) < 2:
            print("Usage: generate <num_tokens>")
        else:
            try:
                num_tokens = int(args[1])
                context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Replace with proper context if available
                generated_tokens = model.generate(context, max_new_tokens=num_tokens)
                generated_text = processor.decode(generated_tokens[0].tolist())
                print("Generated text:", generated_text)
            except ValueError:
                print("Invalid number of tokens. Please provide an integer.")
    elif cmd == "exit":
        print("Exiting...")
        break
    else:
        print(f"Unknown command: {cmd}")
