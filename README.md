# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: THILAK RAJ . P

### Register Number: 212224040353

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Deep learning exp/deep ex 1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name:THILAK RAJ . P
# Register Number:212224040353
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (lig. parameters(), lr=0.001)
loss_history = []

# Name:THILAK RAJ . P
# Register Number:212224040353
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range (epochs) :
      optimizer. zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss. backward()
      optimizer.step()
      loss_history.append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(loss_history, columns=["Loss"])

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information

<img width="192" height="238" alt="image" src="https://github.com/user-attachments/assets/c54c2c94-cfd8-43db-a8b1-f1dc0202410d" />


<img width="1536" height="826" alt="nn" src="https://github.com/user-attachments/assets/ddabbcf5-c4fc-41c7-908e-64aae2f5dfd3" />

### OUTPUT
<img width="589" height="455" alt="download" src="https://github.com/user-attachments/assets/40a28155-d7d6-49dc-9ec7-2bb698ab1447" />


### New Sample Data Prediction
<img width="311" height="39" alt="image" src="https://github.com/user-attachments/assets/8b77ecab-ec35-4fa2-b262-9d88313c2d37" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
