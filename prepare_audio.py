import sys
from pathlib import Path
import librosa
import soundfile as sf

def trim_folder(src: Path, dst: Path, seconds: float = 30.0):
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in src.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}]
    files.sort()
    for i, path in enumerate(files, start=1):
        y, sr = librosa.load(path, sr=22050, mono=True, duration=seconds)
        out = dst / f"{i:02d}.wav"
        sf.write(out, y, sr)
        print(f"{path.name} -> {out.name} ({len(y)/sr:.1f} с)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python prepare_audio.py <входная_папка> <выходная_папка>")
        sys.exit(1)
    trim_folder(Path(sys.argv[1]), Path(sys.argv[2]))