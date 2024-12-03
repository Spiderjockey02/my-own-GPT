import torch
from helpers.GPT import GPTLanguageModel
from helpers.Tokenizer import TokenProcessor
from torch.utils.data import DataLoader, TensorDataset

# Initialize TokenProcessor and GPT model
print("Loading Tokenizer and processing tokens...")
processor = TokenProcessor('./data/text.txt')
processor._load_text()
processor._process_text()

# Initialize GPT model with the tokenizer's vocabulary size
model = GPTLanguageModel(vocab_size=processor.vocab_size)

# Prepare data loader
def create_data_loader(processor, batch_size):
    data = torch.tensor(processor.tokenized_data, dtype=torch.long)
    seq_len = 1024
    n_samples = len(data) - seq_len

    inputs = torch.stack([data[i:i + seq_len] for i in range(n_samples)])
    targets = torch.stack([data[i + 1:i + seq_len + 1] for i in range(n_samples)])

    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 8
data_loader = create_data_loader(processor, batch_size=batch_size)

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
        model.save_model("./data/gpt_model.pth")
        print("Model saved to './data/gpt_model.pth'.")
    elif cmd == "load":
        if len(args) < 2:
            print("Usage: load <file_path>")
        else:
            model.load_model(args[1])
            print(f"Model loaded from '{args[1]}'.")
    elif cmd == "train":
        print("Starting training...")
        model.train_model(data_loader, epochs=10)
        print("Training completed.")
    elif cmd == "eval":
        print("Evaluating model...")
        
        model.evaluate(data_loader)
    elif cmd == "generate":
        if len(args) < 2:
            print("Usage: generate <num_tokens>")
        else:
            try:
                num_tokens = int(args[1])
                context = torch.zeros((1, 1), dtype=torch.long)  # Replace with proper context if available
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
