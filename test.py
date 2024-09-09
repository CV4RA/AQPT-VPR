import torch
from models.aqpt import AQPT
from data_loader import get_dataloader

model = AQPT()
model.load_state_dict(torch.load('path/to/saved_model.pth'))

test_loader = get_dataloader('path/to/test_dataset', batch_size=32, shuffle=False)

model.eval()
with torch.no_grad():
    for images, image_paths in test_loader:
        outputs = model(images)        
