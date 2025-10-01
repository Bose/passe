#from huggingface_hub import snapshot_download
from datasets import load_dataset

#snapshot_download(repo_id="Cnam-LMSSC/vibravox", 
#                  local_dir="/home/ubuntu/data/",
#                  repo_type="dataset")

subset = 'speechless_noisy' # speech_clean, speech_noisy, (speechless_clean), speechless_noisy

for split in ['train', 'validation', 'test']:
    print(f"Downloading {subset} {split} split")
    ds = load_dataset('Cnam-LMSSC/vibravox', name=subset, split=split, streaming=False)

    ds.save_to_disk(f"/home/ubuntu/data/vibravox/{subset}/{split}/")