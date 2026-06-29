from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(image_path):
    image = Image.open(image_path)

    exif = image.getexif()

    metadata = {}

    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        metadata[tag] = value

    return metadata


metadata = get_exif("foto.jpg")

for key, value in metadata.items():
    print(f"{key}: {value}")