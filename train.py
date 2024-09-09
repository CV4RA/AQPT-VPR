import torch
import torch.optim as optim
from data_loader import get_dataloader
from models.aqpt import AQPT
from loss import QuadrupletLoss

batch_size = 32
num_epochs = 20
learning_rate = 0.0001

train_loader = get_dataloader('path/to/train_dataset', batch_size=batch_size)

model = AQPT()
criterion = QuadrupletLoss(margin=1.0)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for images, _ in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        anchor, positive, negative1, negative2 = outputs.chunk(4, dim=0)
        loss = criterion(anchor, positive, negative1, negative2)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
