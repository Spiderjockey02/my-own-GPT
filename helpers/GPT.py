import torch
import torch.nn as nn
import torch.optim as optim

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=768, n_layer=12, n_head=12):
        super().__init__()
        self.vocab_size = vocab_size

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(1024, n_embd)

        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd
            ),
            num_layers=n_layer,
        )

        # Final linear layer
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        token_emb = self.token_embedding(x)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

    def train_model(self, data_loader, epochs=10, lr=3e-4):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                print("adsfdfdfs", total_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def evaluate(self, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device
                outputs = self(inputs)
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
                total_loss += loss.item()

        print(f"Evaluation Loss: {total_loss / len(data_loader)}")

    def generate(self, idx, max_new_tokens):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx = idx.to(device)  # Move input tensor to the device
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
