import json, os, soundfile as sf

folder = "custom_data_fixed"

with open("ground_truth.json") as f:
    ground_truth = json.load(f)
# build a lookup keyed by filename only, since ground_truth.json and this script
# may reference the folder with different path prefixes
gt_by_filename = {os.path.basename(k): v for k, v in ground_truth.items()}

with open("manifest.jsonl", "w") as out:
    for f in sorted(os.listdir(folder)):
        if f.endswith(".wav"):
            path = os.path.join(folder, f)
            duration = sf.info(path).duration
            out.write(json.dumps({
                "audio_filepath": os.path.abspath(path),
                "duration": duration,
                "text": gt_by_filename.get(f, "")  # real ground truth now, empty only if missing
            }) + "\n")