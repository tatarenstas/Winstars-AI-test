import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog",
               "elephant", "horse", "sheep", "spider", "squirrel"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load model function
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

#predict image function
def predict_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence_score:.2f}%")

    plt.figure(figsize=(6, 3))
    plt.bar(CLASS_NAMES, probabilities.cpu().squeeze().numpy())
    plt.xticks(rotation=45)
    plt.ylabel("Confidence")
    plt.title(f"Prediction: {predicted_class} ({confidence_score:.2f}%)")
    plt.show()

model = load_model("best_model.pth")
image_path = "animal10/Animals-10/cow/cow (1012).jpeg"
predict_image(image_path, model)
