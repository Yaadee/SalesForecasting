import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_cleaning import load_and_merge_data, handle_missing_values, convert_categorical_to_numeric, extract_date_features
from src.preprocess import feature_engineering, scale_features
from src.lstm_model import SalesDataset, LSTMModel

# Load and preprocess data
df = load_and_merge_data('data/train.csv', 'data/store.csv')
df = handle_missing_values(df)
df = convert_categorical_to_numeric(df)
df = extract_date_features(df)
df = feature_engineering(df)
df, scaler = scale_features(df, ['Sales', 'Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen'])

# Prepare dataset for LSTM
seq_len = 30
dataset = SalesDataset(df.drop(['Sales', 'Date'], axis=1).values, df['Sales'].values, seq_len)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters
input_dim = df.shape[1] - 2  # Minus 'Sales' and 'Date'
hidden_dim = 64
num_layers = 2
output_dim = 1
num_epochs = 50
learning_rate = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
model.train()
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
torch.save(model.state_dict(), f'app/lstm_model_{timestamp}.pth')
