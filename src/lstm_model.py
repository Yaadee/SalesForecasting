import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SalesDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, seq_len=30):
        self.data = data
        self.target = target
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration purposes
    data = torch.randn(1000, 10)  # 1000 samples, 10 features each
    target = torch.randn(1000)  # 1000 target values

    seq_len = 30
    input_dim = data.shape[1]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    dataset = SalesDataset(data, target, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")
