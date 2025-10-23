import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from cpgpt.loss.loss import c_index_loss

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=1000, n_features=10):
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate survival times (positive values)
    times = np.abs(np.random.normal(500, 200, n_samples))
    
    # Generate event indicators (1 for event/death, 0 for censored)
    events = np.random.binomial(1, 0.7, n_samples)
    
    # Combine times and events
    y = np.column_stack((times, events))
    
    return X, y

# Generate data
X, y = generate_synthetic_data(5000)

# Split data into train, validation, and test sets
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = np.split(X, [train_size, train_size + val_size])
y_train, y_val, y_test = np.split(y, [train_size, train_size + val_size])

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the MLP model with PyTorch Lightning
class SurvivalMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final output layer for risk score (scalar value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)  # Output risk scores
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        risk_scores = self(x)
        times = y[:, 0]
        events = y[:, 1]
        
        loss = c_index_loss(risk_scores, times, events)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        risk_scores = self(x)
        times = y[:, 0]
        events = y[:, 1]
        
        loss = c_index_loss(risk_scores, times, events)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        risk_scores = self(x)
        times = y[:, 0]
        events = y[:, 1]
        
        loss = c_index_loss(risk_scores, times, events)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# Initialize model
input_dim = X_train.shape[1]
model = SurvivalMLP(input_dim)

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='survival-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

# Train the model
trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=True
)

trainer.fit(model, train_loader, val_loader)

# Evaluate on test set
test_result = trainer.test(model, test_loader)
print(f"Test loss: {test_result[0]['test_loss']:.4f}")

# For predictions on new data (example)
def predict_risk(model, features):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        risk_scores = model(features_tensor)
        return risk_scores.numpy()

# Example of prediction
if X_test.shape[0] > 0:
    sample_input = X_test[0:5]
    predicted_risks = predict_risk(model, sample_input)
    print("Sample predictions (risk scores):", predicted_risks)