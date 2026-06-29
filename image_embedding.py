import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from sklearn.metrics.pairwise import cosine_similarity

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

THRESHOLD = 0.95

uploaded_photos = [
    "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/10.jpg",
    # "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/13.jpg"
]

new_photo = "/Users/diajeng/Documents/image-detection-newsrintami/demo_similarity/11.jpg"


def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    embedding = model.predict(img, verbose=0)

    return embedding


new_embedding = get_embedding(new_photo)

for photo in uploaded_photos:

    old_embedding = get_embedding(photo)

    similarity = cosine_similarity(
        new_embedding,
        old_embedding
    )[0][0]

    print(f"{new_photo} vs {photo}")
    print(f"Similarity : {similarity:.4f}")

    if similarity >= THRESHOLD:
        print("Foto terlalu mirip!")
        break

else:
    print("Foto valid")