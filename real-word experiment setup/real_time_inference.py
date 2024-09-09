# File: real_time_inference.py
import cv2
import torch
from aqpt_model import AQPTModel
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = AQPTModel()
model.load_state_dict(torch.load('aqpt_pretrained_weights.pth', map_location='cuda'))
model = model.to('cuda')
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Camera feed setup
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = model(input_tensor)

    # Show real-time frame
    cv2.imshow('Real-time Place Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
