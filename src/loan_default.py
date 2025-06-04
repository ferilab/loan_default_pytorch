
# General and ML libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
import sys

    # Libraries required to make the deep learing model using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from preprocess import process_data

    # Add the absolute pass of the package to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


### 1. Load the data
file_path = os.path.join(project_root, "data/")
df = pd.read_csv(file_path + "SBAcase.11.13.17.csv")

### 2. Data Preparation

df = process_data(df)

# 2.5 Split and normalize: We'll keep part of data for validation and another part for the final testing

# Separate features and target
X = df.drop(columns=["MIS_Status"]).values
y = df["MIS_Status"].values

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2.6 Scale using training set only

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

### 3. Deep learning model and training with PyTorch

# 3.1 Convert the data to tensors and batch them. 
# Note: y_train (and _val, _test) are numpy array of type int64 so we need to explicitly cast them to torch.long as 
# torch might fail to infer the correcttensor type automatically. 

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# batch the datasets to be fed to the model
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)

# 3.2 This is our MLP model. We'll make a child class of Torch's nn and will configure it. 
# Let's have 3 layers in total: the input layer of the batch size, a hidden layer, and the binary output layer.


class LoanNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.model(x)

    # Make an instance of the model
model = LoanNet(X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3.3 Let's train the model instance 

print("n\Training the model:")
print("-" * 50)

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
    acc = correct / len(val_ds)
    print(f"Epoch {epoch+1}: Loss {total_loss:.3f}, Val Acc {acc:.3f}")

# 4. Evaluation of the model

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds = model(xb).argmax(1)
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.tolist())

print("\nclassification report:")
print("-" * 50)
print(classification_report(all_labels, all_preds, target_names=["CHGOFF", "PIF"]))