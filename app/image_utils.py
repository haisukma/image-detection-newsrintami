from PIL import Image


def crop_bbox(image_path, bbox):

    img = Image.open(image_path)

    x1, y1, x2, y2 = bbox

    return img.crop((x1, y1, x2, y2))