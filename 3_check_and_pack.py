"""Шаг 3. Проверка разметки и сборка архива YOLOv8.

- у каждого фото в архиве будет свой .txt, для фото без объектов — пустой;
- номера классов в пределах списка, координаты от 0 до 1, рамки не нулевые и не за краем;
- классы в разметке совпадают с исходным списком (LabelImg не дописал новых);
- в фото не осталось EXIF-поворота;
- рисует превью с рамками в work/preview, чтобы глазами проверить, что рамки на объектах;
- собирает архив YOLOv8: train/images, train/labels, data.yaml.

Запуск:
    python 3_check_and_pack.py work dataset.zip
"""
import sys
import warnings
import zipfile
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)  # PIL ругается на обрезанный EXIF у части фото

from PIL import Image, ImageDraw, ImageFont

COLORS = ["#E53935", "#FB8C00", "#1E88E5", "#3949AB", "#00897B"]


def load_classes(work):
    path = work / "labels" / "classes.txt"
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def data_yaml(classes):
    return ("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
            f"nc: {len(classes)}\nnames: {classes}\n")


def check_labels(work, CLASSES):
    images, labels = work / "images", work / "labels"
    problems, created_empty, stats, empties = [], [], Counter(), 0

    predefined = work / "predefined_classes.txt"
    if predefined.exists():
        order = [l.strip() for l in predefined.read_text(encoding="utf-8").splitlines() if l.strip()]
        if order != CLASSES:
            problems.append(f"классы в разметке {CLASSES} не совпадают с исходным списком {order}: номера классов съехали")

    for img_path in sorted(images.glob("*.jpg")):
        txt = labels / (img_path.stem + ".txt")
        if not txt.exists():
            created_empty.append(img_path.name)  # в архив пойдёт пустой .txt, рабочую папку не трогаем
        img = Image.open(img_path)
        if img.getexif().get(274, 1) not in (1, None):
            problems.append(f"{img_path.name}: в фото остался EXIF-поворот, прогони шаг 1")
        content = txt.read_text(encoding="utf-8") if txt.exists() else ""
        lines = [l.split() for l in content.splitlines() if l.strip()]
        if not lines:
            empties += 1
        boxes = []
        for n, parts in enumerate(lines, 1):
            where = f"{txt.name}, строка {n}"
            if len(parts) != 5:
                problems.append(f"{where}: ожидалось 5 чисел, а их {len(parts)}")
                continue
            try:
                c = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(f"{where}: не числа")
                continue
            if not 0 <= c < len(CLASSES):
                problems.append(f"{where}: класс {c} вне 0–{len(CLASSES) - 1}")
            if w <= 0 or h <= 0:
                problems.append(f"{where}: нулевая рамка")
            if min(x - w / 2, y - h / 2) < -0.002 or max(x + w / 2, y + h / 2) > 1.002:
                problems.append(f"{where}: рамка выходит за край фото")
            for c2, x2, y2, w2, h2 in boxes:
                if c2 == c and abs(x - x2) < 0.003 and abs(y - y2) < 0.003 and abs(w - w2) < 0.003 and abs(h - h2) < 0.003:
                    problems.append(f"{where}: дубль рамки")
            boxes.append((c, x, y, w, h))
            if 0 <= c < len(CLASSES):
                stats[CLASSES[c]] += 1
    return problems, created_empty, stats, empties


def draw_previews(work, CLASSES):
    out = work / "preview"
    out.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for img_path in sorted((work / "images").glob("*.jpg")):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        txt = work / "labels" / (img_path.stem + ".txt")
        content = txt.read_text(encoding="utf-8") if txt.exists() else ""
        for line in content.splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            box = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
            color = COLORS[c % len(COLORS)]
            d.rectangle(box, outline=color, width=3)
            d.text((box[0] + 2, max(0, box[1] - 16)), CLASSES[c] if c < len(CLASSES) else str(c), fill=color, font=font)
        img.thumbnail((1280, 1280))
        img.save(out / img_path.name, quality=85)


def pack(work, zip_path, CLASSES):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img_path in sorted((work / "images").glob("*.jpg")):
            z.write(img_path, f"train/images/{img_path.name}")
            txt = work / "labels" / (img_path.stem + ".txt")
            z.writestr(f"train/labels/{img_path.stem}.txt", txt.read_text(encoding="utf-8") if txt.exists() else "")
        z.writestr("data.yaml", data_yaml(CLASSES))
        z.writestr("classes.txt", "\n".join(CLASSES) + "\n")


def main(work, zip_path):
    work = Path(work)
    CLASSES = load_classes(work)
    problems, created_empty, stats, empties = check_labels(work, CLASSES)
    n_images = len(list((work / "images").glob("*.jpg")))
    print(f"Фото: {n_images}, рамок: {sum(stats.values())}, пустых фото: {empties}")
    for c in CLASSES:
        print(f"  {c}: {stats[c]}")
    if created_empty:
        print(f"\nФото без разметки: {len(created_empty)}. В архив они пойдут с пустым .txt. Убедись, что на них правда нет нужных объектов:")
        for name in created_empty:
            print("  ", name)
    draw_previews(work, CLASSES)
    print(f"\nПревью с рамками: {work / 'preview'}")
    if problems:
        print(f"\nНайдены проблемы ({len(problems)}), архив не собран:")
        for p in problems:
            print("  ", p)
        sys.exit(1)
    pack(work, zip_path, CLASSES)
    print(f"\nОшибок нет. Архив готов: {zip_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
