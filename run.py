import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0],
                  [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)

y = torch.tensor([[30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0],
                  [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(1, 5)
    self.fc2 = nn.Linear(5, 1)

  def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x
  
model = Net()

criteria = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x)
    loss = criteria(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 99:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Inference
with torch.no_grad():
    predition = model(torch.tensor([[10.0]], dtype=torch.float32))
    print(f'Previsão de tempo de conclusão: {predition.item()} minutos')

# Save the model
torch.save(model, 'model.pth')

# Save the model's state dictionary
torch.save(model.state_dict(), 'model_state_dict.pth')

# # Load the model
# model = torch.load('model.pth')

# # Load the model's state dictionary
# model = Net()
# model.load_state_dict(torch.load('model_state_dict.pth'))
# model.eval()
