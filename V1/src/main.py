import torch as torch
import torch.nn as nn
import torch.optim as optim

# Print Device in Use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  
        self.fc2 = nn.Linear(4, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)               
        return x

# Model Trainer
X_train = torch.rand(4, 2)
y_train = torch.rand(4, 1)
# Training Data
print("Training Data:")
print(X_train)
print(y_train)
print()

Old_X_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) 
Old_y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Instantiate the Model, Define Loss Function and Optimizer
model = NeuralNetwork
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Model Trainer
for epoch in range(100):  
    model.train() 

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)  
    
    # Backward pass and optimize
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
        print("Current GPU memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    
    # Evaluate The Model
    model.eval()  
with torch.no_grad(): 
    test_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    predictions = model(test_data) 
    print()
    print(f'Predictions:\n{predictions}')
return predictions