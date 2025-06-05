from PIL import Image, ImageDraw

def detect_and_blur_faces(image: Image.Image) -> Image.Image:
    # Просто рисуем квадрат — заглушка
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([w*0.3, h*0.3, w*0.7, h*0.7], fill="gray", outline="red", width=5)
    return img