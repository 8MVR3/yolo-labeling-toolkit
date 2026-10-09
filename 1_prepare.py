"""Шаг 1. Подготовка фото к разметке.

Поворачивает фото по метке EXIF (как Auto-Orient в Roboflow) и сохраняет
их уже без этой метки. Иначе LabelImg покажет фото боком, а после импорта
в Roboflow рамки съедут с объектов.

Запуск:
    python 1_prepare.py папка_с_фото work classes.txt

classes.txt — список классов по одному в строке, в том же порядке, что в data.yaml.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)  # PIL ругается на обрезанный EXIF у части фото

from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def main(src, work, classes_file):
    src, work = Path(src), Path(work)
    classes = [l.strip() for l in Path(classes_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    images_dir, labels_dir = work / "images", work / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rotated, total = [], 0
    for path in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT):
        img = Image.open(path)
        orientation = img.getexif().get(274, 1)
        fixed = ImageOps.exif_transpose(img).convert("RGB")
        out = images_dir / (path.stem + ".jpg")
        fixed.save(out, "JPEG", quality=95)  # сохраняем без EXIF, поворот уже в пикселях
        total += 1
        if orientation not in (1, None):
            rotated.append(f"{path.name} (метка {orientation})")

    # Порядок классов строго как в data.yaml: от него зависят номера классов в разметке.
    (work / "predefined_classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    print(f"Готово: {total} фото в {images_dir}")
    print(f"Повёрнуто по EXIF: {len(rotated)}")
    for r in rotated:
        print("  ", r)
    print(f"\nКлассы для LabelImg: {work / 'predefined_classes.txt'}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
