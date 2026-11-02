from huggingface_hub import hf_hub_download
import os
import shutil

repo_id = "K-areem/AI-Interview-Questions"

save_dir = "dataset/interview_questions"
os.makedirs(save_dir, exist_ok=True)

files = [
    "data/train-00000-of-00001.parquet",
    "data/eval-00000-of-00001.parquet"
]

for file in files:
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=file
    )

    shutil.copy2(
        downloaded,
        os.path.join(save_dir, os.path.basename(file))
    )

    print("Downloaded:", file)

print("All downloads completed!")