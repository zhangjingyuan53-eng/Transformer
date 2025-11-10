import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

#  data loading
X_data = np.load()
Y_data = np.load()

# normalization processing
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_data_scaled = scaler_x.fit_transform(X_data)
y_data_scaled = scaler_y.fit_transform(Y_data)

# Data division: 60% for training, 20% for validation, and 20% for testing
indices = np.arange(X_data.shape[0])
x_temp, x_test, y_temp, y_test, time_temp, time_test, idx_temp, idx_test = train_test_split(
    X_data_scaled, y_data_scaled, indices, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val, _, _, idx_train, idx_val = train_test_split(
    x_temp, y_temp, time_temp, idx_temp, test_size=0.25)

# Increase the channel dimension (B, 1, L)
x_train = np.expand_dims(x_train, axis=1)
x_val = np.expand_dims(x_val, axis=1)
x_test = np.expand_dims(x_test, axis=1)

# to Tensor
x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)
x_val_tensor = torch.Tensor(x_val)
y_val_tensor = torch.Tensor(y_val)
x_test_tensor = torch.Tensor(x_test)
y_test_tensor = torch.Tensor(y_test)

# Build batch data
batch_size = 32
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#  Definition of Transformer regression model
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, num_layers, output_dim, dropout=0.2):
        super(TransformerRegressor, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim // 2, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)           # (B, 1, input_dim) -> (B, 1, emb_dim)
        x = x.permute(1, 0, 2)           # (seq_len=1, batch, emb_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)                # (batch, emb_dim)
        out = torch.sigmoid(self.output_layer(x))  # (batch, output_dim)
        return out

input_dim = x_train_tensor.shape[2]
emb_dim = 128
num_heads = 4
num_layers = 2
output_dim = y_train_tensor.shape[1]

model = TransformerRegressor(input_dim, emb_dim, num_heads, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Early Stopping
best_val_loss = float('inf')
patience = 30
counter = 0
best_model_path = r"\trans.pth"

# model training
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation set evaluation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_outputs = model(xb)
            loss = criterion(val_outputs, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model at epoch {epoch+1}")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Test set evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    predictions = model(x_test_tensor)

# Normalized albedo
y_test_denorm = scaler_y.inverse_transform(y_test_tensor.numpy())
predictions_denorm = scaler_y.inverse_transform(predictions.numpy())

#  RMSE
for i in range(output_dim):
    rmse = np.sqrt(mean_squared_error(y_test_denorm[:, i], predictions_denorm[:, i]))
    print(f"variable{i+1}: {rmse:.4f}")



