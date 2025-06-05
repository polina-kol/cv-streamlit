import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# Загрузка классификационной модели EfficientNet-B0
classification_model = torch.load("models/classifier.pth", map_location="cpu")
classification_model.eval()

# YOLOv5s модели по срезам
yolo_models = {
    "axial": YOLO("models/yolo_axial.pt"),
    "sagittal": YOLO("models/yolo_sag.pt"),
    "coronal": YOLO("models/yolo_coronal.pt"),
}

# Преобразование для классификации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Классы типов срезов (в том же порядке, как обучалась классификация)
slice_classes = ["axial", "coronal", "sagittal"]

def classify_slice(image: Image.Image) -> str:
    """Классифицирует тип среза мозга (axial/sagittal/coronal)."""
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        output = classification_model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return slice_classes[predicted]

def detect_tumor(image: Image.Image, slice_type: str) -> Image.Image:
    """Детектирует опухоль на изображении с помощью соответствующей YOLO модели."""
    model = yolo_models.get(slice_type.lower())
    if model is None:
        raise ValueError(f"No YOLO model found for slice type: {slice_type}")

    results = model.predict(image, conf=0.25, save=False, imgsz=512)
    annotated_image = results[0].plot()  # Получаем изображение с аннотациями
    return Image.fromarray(annotated_image)
