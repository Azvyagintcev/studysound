from huggingface_hub import snapshot_download

print("Начинаю скачивание модели...")
snapshot_download(
    repo_id="intfloat/multilingual-e5-small",
    local_dir="./model_e5",
    local_dir_use_symlinks=False
)
print("Готово! Модель скачана в папку model_e5")