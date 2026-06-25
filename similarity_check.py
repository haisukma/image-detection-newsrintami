from PIL import Image
import imagehash

THRESHOLD = 15

uploaded_photos = [
    "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/12.jpg",
    "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/13.jpg"
]

new_photo = "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/14.jpg"

new_hash = imagehash.phash(Image.open(new_photo))

for photo in uploaded_photos:
    old_hash = imagehash.phash(Image.open(photo))

    distance = new_hash - old_hash

    print(f"{new_photo} vs {photo}: {distance}")

    if distance <= THRESHOLD:
        print("Foto terlalu mirip!")
        break
else:
    print("Foto valid")