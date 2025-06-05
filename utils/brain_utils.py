def classify_slice(image):
    # Заглушка — всегда axial
    return "axial"

def detect_tumor(image, slice_type):
    from PIL import ImageDraw
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.ellipse([w*0.4, h*0.4, w*0.6, h*0.6], outline="red", width=5)
    return img