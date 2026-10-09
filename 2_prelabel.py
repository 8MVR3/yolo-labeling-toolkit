"""Шаг 2 (по желанию). Предразметка моделью.

Обучает YOLO на уже размеченных фото и размечает ею остальные.
Готовую ручную разметку не трогает: пишет только файлы для фото без .txt.
Список предразмеченных фото сохраняет в prelabeled.txt: каждое из них нужно
открыть в LabelImg и проверить рамки и классы.

Нужно: pip install ultralytics  (видеокарта подхватится сама)

Запуск:
    python 2_prelabel.py work
    python 2_prelabel.py work --epochs 80 --imgsz 1280 --batch 4 --conf 0.35
    python 2_prelabel.py work --dry-run      # только собрать датасет, без обучения
"""
import argparse
import random
import shutil
from pathlib import Path



def build_dataset(work, ds):
    images, labels = work / "images", work / "labels"
    classes = [l.strip() for l in (labels / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    labeled = sorted(p for p in images.glob("*.jpg") if (labels / (p.stem + ".txt")).exists())
    unlabeled = sorted(p for p in images.glob("*.jpg") if not (labels / (p.stem + ".txt")).exists())
    if len(labeled) < 30:
        raise SystemExit(f"Размечено всего {len(labeled)} фото. Для предразметки нужно хотя бы 30, лучше 80–100.")

    random.seed(0)
    shuffled = labeled[:]
    random.shuffle(shuffled)
    n_val = max(5, len(shuffled) // 5)
    splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}

    if ds.exists():
        shutil.rmtree(ds)
    for split, items in splits.items():
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir(parents=True)
        for img in items:
            shutil.copy2(img, ds / split / "images" / img.name)
            shutil.copy2(labels / (img.stem + ".txt"), ds / split / "labels" / (img.stem + ".txt"))

    yaml = ds / "data.yaml"
    yaml.write_text(
        f"path: {ds.resolve().as_posix()}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(classes)}\nnames: {classes}\n", encoding="utf-8")
    print(f"Датасет: обучение {len(splits['train'])}, проверка {len(splits['val'])}, без разметки {len(unlabeled)}")
    return yaml, unlabeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    work = Path(a.work)
    yaml, unlabeled = build_dataset(work, work / "_dataset")
    if a.dry_run:
        print(f"Проверка без обучения: {yaml}")
        return
    if not unlabeled:
        print("Все фото уже размечены, предразметка не нужна.")
        return

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                project=str(work / "_runs"), name="prelabel", exist_ok=True, verbose=False)
    best = YOLO(str(model.trainer.best))

    labels = work / "labels"
    done = []
    for res in best.predict(source=[str(p) for p in unlabeled], conf=a.conf, imgsz=a.imgsz,
                            stream=True, verbose=False):
        stem = Path(res.path).stem
        lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for (x, y, w, h), c in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist())]
        (labels / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        done.append(stem + ".jpg")

    (work / "prelabeled.txt").write_text("\n".join(done) + "\n", encoding="utf-8")
    print(f"Предразмечено {len(done)} фото. Список в {work / 'prelabeled.txt'}.")
    print("Открой их в LabelImg и проверь каждую рамку и класс.")


if __name__ == "__main__":
    main()
