import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import librosa
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA = Path("data")
CSV = DATA / "tracks.csv"


def build_doc(row) -> str:
    return (f"Жанр: {row['genre']}. Настроение: {row['mood']}. "
            f"Вокал: {row['vocal']}. Энергия: {row['energy_1_5']}/5. "
            f"Подходит для: {row['suitability']}. {row['description_ru']}")


def main():
    if not CSV.exists():
        print(f"ОШИБКА: Файл {CSV} не найден. Положите tracks.csv в папку data/")
        return

    df = pd.read_csv(CSV, sep=';')
    df = df.fillna("")
    print(f"Загружено треков: {len(df)}")

    print("Считаю текстовые эмбеддинги (скачиваю модель ~120 МБ)...")
    model = SentenceTransformer("./model_e5", device="cpu", local_files_only=True)
    print(f"Пример текста для нейросети: {build_doc(df.iloc[0])}")
    docs = df.apply(build_doc, axis=1).tolist()
    embs = model.encode(docs, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    np.save(DATA / "embeddings.npy", embs)

    print("Считаю аудио-фичи...")
    rows = []
    for _, row in df.iterrows():
        path = DATA / row["file_path"]
        if not path.exists():
            print(f"[!] Нет файла: {path}")
            rows.append({"bpm": row["bpm_target"], "rms": 0.0, "centroid": 0.0})
            continue
        y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        rms = float(np.mean(librosa.feature.rms(y=y)[0]))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rows.append({"bpm": bpm, "rms": rms, "centroid": centroid})

    pd.DataFrame(rows).to_csv(DATA / "audio_features.csv", index=False)
    print("Готово!")


if __name__ == "__main__":
    main()