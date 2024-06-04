import torch
import torch.nn as nn

# Create sample values
predicted = torch.tensor([[2.5, 4.8, 6.9, 9.5], [2.5, 4.8, 6.9, 9.5]])
actual = torch.tensor([[3.0, 5.0, 7.0, 9.0], [3.0, 5.0, 7.0, 9.0]])


print(predicted.shape, actual.shape)
# Create and use criterion
criterion = nn.MSELoss()
loss = criterion(predicted, actual)

print(f'MSE Loss: {loss}')

# Returns: MSE Loss: 0.13749998807907104