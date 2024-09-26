# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# from model import NbrilNet

# train_data = np.load('C:/Users/mseok/VSCode/Challenge/trainset.npy')  
# train_labels = np.load('C:/Users/mseok/VSCode/Challenge/trainlabel.npy')  
# test_data = np.load('C:/Users/mseok/VSCode/Challenge/testset.npy')

# train_data = np.transpose(train_data, (0, 3, 1, 2))  # (샘플 수, 채널 수, 높이, 너비)
# test_data = np.transpose(test_data, (0, 3, 1, 2))    # (샘플 수, 채널 수, 높이, 너비)

# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

# X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.1)

# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)  
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long)
# X_test_tensor = torch.tensor(test_data, dtype=torch.float32)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

# X_train_tensor = X_train_tensor.to(device)
# y_train_tensor = y_train_tensor.to(device)
# X_val_tensor = X_val_tensor.to(device)
# y_val_tensor = y_val_tensor.to(device)
# X_test_tensor = X_test_tensor.to(device)

# #SE block + non local neural net + custom dataset

# model = NbrilNet().to(device)

# criterion = nn.CrossEntropyLoss()  
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10
# batch_size = 64

# train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# for epoch in range(num_epochs):
#     model.train()  
#     running_loss = 0.0
#     correct_preds = 0
#     total_preds = 0

#     for X_batch, y_batch in train_loader:
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         _, predicted = torch.max(outputs, 1)
#         correct_preds += (predicted == y_batch).sum().item()
#         total_preds += y_batch.size(0)

#     epoch_loss = running_loss / len(train_loader)
#     epoch_accuracy = correct_preds / total_preds

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}")

# model.eval()  
# correct_preds = 0
# total_preds = 0
# with torch.no_grad():
#     for X_batch, y_batch in val_loader:
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         correct_preds += (predicted == y_batch).sum().item()
#         total_preds += y_batch.size(0)

# val_accuracy = correct_preds / total_preds
# print(f"Validation Accuracy: {val_accuracy:.2f}")

# with torch.no_grad():
#     test_outputs = model(X_test_tensor)
#     _, test_predicted = torch.max(test_outputs, 1)
#     test_predicted = test_predicted.cpu().numpy()

# import pandas as pd
# submission = pd.read_csv('sample_submission.csv')
# submission['label'] = test_predicted
# submission.to_csv('submission.csv', index=False)

# print('Submission file created: submission.csv')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from model import NbrilNet  # Assuming your model is saved in model.py
from dataset import CustomDataset  # Assuming your dataset is saved in dataset.py

# Load the datasets
train_data = np.load('C:/Users/mseok/VSCode/Challenge/trainset.npy')
train_labels = np.load('C:/Users/mseok/VSCode/Challenge/trainlabel.npy')
test_data = np.load('C:/Users/mseok/VSCode/Challenge/testset.npy')

# Split train data into train and validation sets (90% train, 10% validation)
train_idx, val_idx = train_test_split(np.arange(len(train_data)), test_size=0.1, random_state=42, stratify=train_labels)
train_dataset = CustomDataset(images=train_data[train_idx], labels=train_labels[train_idx], mode='train')
val_dataset = CustomDataset(images=train_data[val_idx], labels=train_labels[val_idx], mode='test')  # Validation set

# Define test dataset (test set labels assumed to be unavailable)
test_dataset = CustomDataset(images=test_data, labels=None, mode='test')

# Define data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define some parameters
epochs = 20
learning_rate = 0.001

# Define the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = NbrilNet().to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

    accuracy = total_correct / total_samples
    print(f"Training Accuracy: {accuracy:.2f}%")

# Validation loop with accuracy calculation
def validate(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in val_loader:
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = total_correct / total_samples
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return accuracy

# Training process
best_val_accuracy = 0.0
for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    val_accuracy = validate(model, val_loader, criterion)

    # Save model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model

print(f"Training Finished! Best Validation Accuracy: {best_val_accuracy:.2f}%")

# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Generate predictions for the test set and create a submission file
X_test_tensor = torch.Tensor(test_data).permute(0, 3, 1, 2)  # Convert test data to the appropriate tensor shape

model.eval()  # Set model to evaluation mode for test inference
with torch.no_grad():
    test_outputs = model(X_test_tensor.to('cuda' if torch.cuda.is_available() else 'cpu'))
    _, test_predicted = torch.max(test_outputs, 1)
    test_predicted = test_predicted.cpu().numpy()

# Create submission file
submission = pd.read_csv('sample_submission.csv')
submission['label'] = test_predicted
submission.to_csv('submission.csv', index=False)

print('Submission file created: submission.csv')
