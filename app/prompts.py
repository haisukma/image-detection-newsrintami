CONDITIONS = {
    "bracing": [
        "Normal",
        "Korosi",
        "Bengkok",
        "Hilang"
    ],
    "jumper": [
        "Normal",
        "Korosi",
        "Rantas",
        "Putus",
        "Lepas",
        "Hilang"
    ],
    "insulator": [
        "Normal",
        "Retak Rambut",
        "Gumpil",
        "Pecah",
        "Flashover"
    ],
    "aksesoris sisi cold": [
        "Normal",
        "Korosi"
    ],
    "aksesoris sisi hot": [
        "Normal",
        "Korosi"
    ]
}

def build_prompt(item_name):

    item_name = item_name.lower()

    if item_name not in CONDITIONS:
        raise ValueError(f"{item_name} tidak dikenali")

    kondisi = "\n".join(
        f"- {c}" for c in CONDITIONS[item_name]
    )

    return f"""
Anda adalah seorang inspektor tower transmisi tenaga listrik yang berpengalaman.

YOLO telah mendeteksi bahwa objek pada gambar adalah:

{item_name}

Analisis kondisi objek berdasarkan gambar.

Anda HANYA boleh memilih SATU kondisi berikut:

{kondisi}

Jangan memilih kondisi yang tidak ada pada daftar.

Jika objek tidak terlihat jelas atau tidak dapat dinilai, jawab:

Kondisi: Tidak Dapat Dinilai

Jawaban HARUS menggunakan format:

Item: {item_name}
Kondisi:
Confidence:
"""