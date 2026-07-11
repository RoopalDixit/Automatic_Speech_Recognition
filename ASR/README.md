# Streaming ASR + Barge-In Engine — Project README

A portfolio project building a real-time, interruptible speech-to-text pipeline
tuned for gaming voice chat, using NVIDIA NeMo's cache-aware streaming ASR models.
This README documents progress through **Step 5 (VAD)** of the build.

---

## 1. Core ASR Concepts Used in This Project

Before diving into the build log, here's the conceptual foundation this project
exercises — useful both as orientation and as an interview reference sheet.

**Automatic Speech Recognition (ASR)** — converting spoken audio into text. Modern
neural ASR systems typically pair an *encoder* (which turns audio into a sequence of
learned representations) with a *decoder* (which turns those representations into
text tokens).

**Streaming vs. offline inference** — offline ASR processes a complete audio file at
once, with full access to past and future context. Streaming ASR processes audio
incrementally, in small chunks, as it arrives — which is what any live voice product
requires, but which sacrifices some accuracy since the model can't see future audio
when deciding what a word is.

**Cache-aware streaming** — rather than reprocessing all prior audio on every new
chunk (prohibitively expensive), the model maintains an internal cache (analogous to
KV-cache in LLMs) of previously computed intermediate representations, so each new
chunk only requires incremental computation.

**Chunk size / shift size** — the amount of audio the model consumes per streaming
step. Smaller chunks reduce latency (you get partial transcripts faster) but reduce
accuracy (less context per decision). This is a direct, tunable latency-vs-accuracy
dial, and one we measured empirically in this project (see Step 4 below).

**Left context / left chunks** — how much previously-processed audio the model is
allowed to reference when processing the current chunk. More left context generally
improves accuracy at some memory/compute cost.

**TDT (Token-and-Duration Transducer)** — the decoding architecture used by NVIDIA's
Parakeet models. Unlike standard RNN-T decoders which predict one output per input
frame, TDT jointly predicts a token *and* how many frames to skip, which reduces
redundant computation and speeds up decoding — directly relevant to low-latency
streaming.

**FastConformer** — the encoder backbone behind Parakeet, a convolution-augmented
transformer architecture optimized for speed relative to a standard Conformer, while
retaining most of its accuracy.

**Word Error Rate (WER)** — the standard ASR accuracy metric: the number of word
substitutions, insertions, and deletions needed to turn the model's output into the
ground truth, divided by the number of words in the ground truth. Lower is better.

**Domain-specific evaluation** — general ASR benchmarks (LibriSpeech, Open ASR
Leaderboard) don't reflect performance on specialized vocabulary. This project builds
a custom gaming-vocabulary evaluation set (callouts, ability names, slang) since
that's the domain a gaming voice AI product actually needs to handle well.

**Voice Activity Detection (VAD)** — a lightweight model or heuristic that decides
whether a given audio frame contains speech, used to gate more expensive ASR
processing and to detect turn-taking boundaries. This is Step 5 of the project (in
progress — see "What's Next" below).

**Barge-in** — the ability for a system to detect that a user has started speaking
*while the AI companion is still talking*, and interrupt playback accordingly. This
is the signature capability the whole project is building toward (Step 6, not yet
started).

---

## 2. Step-by-Step Implementation Guide

This section gives the **clean, corrected command sequence** for reproducing Steps
1–5 — i.e., what to actually run, incorporating every fix from the error log in
Section 3. Each step also states *what* is being implemented and *why*, not just the
commands.

### Step 1 — Environment Setup

**What's being implemented:** An isolated Python environment so NeMo's dependency
tree (PyTorch, audio libraries, etc.) doesn't collide with anything else on the
machine.

```bash
# Install Miniconda first if you don't have it (macOS example, Apple Silicon):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
# restart your terminal, then confirm:
conda --version

# If you hit a Terms-of-Service error on `conda create`, accept it once per machine:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate the project environment:
conda create -n asr python=3.10 -y
conda activate asr   # do this again every time you open a new terminal

# Install PyTorch (CPU build shown; swap the index URL for a CUDA build if you have a GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install NeMo — note the quotes, required on zsh (macOS default shell):
pip install "nemo_toolkit[asr]" --break-system-packages

# Supporting libraries:
pip install soundfile librosa sounddevice numpy jiwer --break-system-packages
```

### Step 2 — Offline Transcription Sanity Check

**What's being implemented:** A one-shot (non-streaming) transcription call, to
confirm the whole environment and model download pipeline works before adding
real-time complexity.

```python
# sanity_check.py
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
transcript = model.transcribe(["test.wav"])
print(transcript[0].text)   # .text pulls the clean string out of the Hypothesis object
```
```bash
conda activate asr          # make sure the prompt shows (asr), not (base)
python3 sanity_check.py
```

### Step 3 — Gaming-Vocabulary Eval Set + WER Benchmarking

**What's being implemented:** A domain-specific accuracy benchmark — since public
ASR benchmarks don't reflect performance on gaming callouts/slang — run across three
different models for comparison.

```bash
# 1. Record ~50 short gaming-vocab clips into custom_data/, named clip_001.wav, clip_002.wav, ...
#    (see the 50-phrase starter list in the walkthrough doc if you need prompts)

# 2. Verify the files are genuinely WAV (not AAC mislabeled as .wav):
file custom_data/clip_001.wav
# If it reports "MPEG-4"/"AAC" instead of "WAVE audio", convert first:
mkdir custom_data_fixed
for f in custom_data/*.wav; do
  filename=$(basename "$f")
  ffmpeg -i "$f" -ar 16000 -ac 1 -c:a pcm_s16le "custom_data_fixed/$filename"
done
```
```python
# build_ground_truth.py — generates the ground_truth.json skeleton from real files
import os, json

folder = "custom_data_fixed"
skeleton = {
    os.path.join(folder, f): ""   # fill in each correct transcript by hand afterward
    for f in sorted(os.listdir(folder))
    if f.endswith(".wav")
}
with open("ground_truth.json", "w") as out:
    json.dump(skeleton, out, indent=2)
```
```bash
python3 build_ground_truth.py
# then open ground_truth.json and type in the correct transcript for each clip
```
```python
# run_parakeet_wer.py
import json
import nemo.collections.asr as nemo_asr
from jiwer import wer

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")

with open("ground_truth.json") as f:
    ground_truth = json.load(f)

predictions = {}
for filename in ground_truth:
    result = model.transcribe([filename])
    predictions[filename] = result[0].text

total_wer = wer(list(ground_truth.values()), list(predictions.values()))
print(f"Parakeet WER on gaming vocab set: {total_wer:.2%}")
```
```bash
pip install openai-whisper moonshine-onnx --break-system-packages
python3 run_parakeet_wer.py
```
```python
# run_whisper_wer.py — do NOT name this file whisper.py, it will shadow the real package
import json, whisper
from jiwer import wer

whisper_model = whisper.load_model("base")

with open("ground_truth.json") as f:
    ground_truth = json.load(f)

predictions = {}
for filename in ground_truth:
    result = whisper_model.transcribe(filename)
    predictions[filename] = result["text"].strip()

total_wer = wer(list(ground_truth.values()), list(predictions.values()))
print(f"Whisper WER on gaming vocab set: {total_wer:.2%}")
```
```bash
python3 run_whisper_wer.py
```

### Step 4 — Cache-Aware Streaming Inference

**What's being implemented:** Converting from one-shot transcription to real
chunk-by-chunk streaming inference — the actual technical core of the project, since
this is what a live voice product needs.

```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo/examples/asr/asr_cache_aware_streaming
pip install hydra-core --break-system-packages
```
```python
# generating_manifest.py — builds the JSONL manifest NeMo's streaming script expects,
# with real ground truth folded in directly
import json, os, soundfile as sf

folder = "/full/path/to/custom_data_fixed"   # use your real absolute path

with open("/full/path/to/ground_truth.json") as f:
    ground_truth = json.load(f)
gt_by_filename = {os.path.basename(k): v for k, v in ground_truth.items()}

with open("manifest.jsonl", "w") as out:
    for f in sorted(os.listdir(folder)):
        if f.endswith(".wav"):
            path = os.path.join(folder, f)
            duration = sf.info(path).duration
            out.write(json.dumps({
                "audio_filepath": os.path.abspath(path),
                "duration": duration,
                "text": gt_by_filename.get(f, "")
            }) + "\n")
```
```bash
python3 generating_manifest.py

# Check the real Hydra config schema for your installed NeMo version — run --help ALONE:
python speech_to_text_cache_aware_streaming_infer.py --help

# Then run streaming inference (no dashes on config keys — this is Hydra, not argparse):
python speech_to_text_cache_aware_streaming_infer.py \
    pretrained_name=nvidia/parakeet-tdt_ctc-110m \
    dataset_manifest=/full/path/to/manifest.jsonl \
    chunk_size=16 \
    shift_size=16 \
    left_chunks=2

# Follow-up comparison: bigger model + bigger chunk size, to see how much WER improves
python speech_to_text_cache_aware_streaming_infer.py \
    pretrained_name=nvidia/parakeet-tdt-1.1b \
    dataset_manifest=/full/path/to/manifest.jsonl \
    chunk_size=64 \
    shift_size=64 \
    left_chunks=2
```
This prints a `CacheAwareStreamingConfig(...)` line confirming the true streaming
path is active, streams the manifest's clips in batches, and prints a real
`WER% of streaming mode: ...` once `text` is populated in the manifest.

### Step 5 — Voice Activity Detection (VAD)

**What's being implemented:** A lightweight speech/no-speech gate that sits in front
of the (expensive) ASR model, so audio is only forwarded to ASR when someone is
actually talking — this also lays the groundwork for turn-taking and barge-in in
Step 6.

```bash
pip install silero-vad sounddevice --break-system-packages
```
```python
# vad_test.py
import torch

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, read_audio, _, _) = utils

# Run VAD over one of your existing test clips first, before wiring up live mic input:
wav = read_audio('custom_data_fixed/clip_001.wav', sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
print(speech_timestamps)
```
```bash
python3 vad_test.py
```
Expected output: a list of `{'start': ..., 'end': ...}` sample-index ranges marking
where speech was detected in the clip — confirming VAD is correctly distinguishing
speech from silence before moving on to live microphone input and turn-taking logic.

---

## 3. Progress Log — Errors, Fixes, and What We Learned

This section documents the actual build history: what was attempted, every real
error encountered, and how each was resolved.

### Step 1 — Environment Setup
with Python 3.10, installed PyTorch (CPU build), then installed NVIDIA NeMo's ASR
toolkit.

**Errors hit and how we fixed them:**

| Error | Cause | Fix |
|---|---|---|
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | Newer conda versions (26.x) require explicit ToS acceptance for Anaconda's default channels before creating an environment | Ran `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` and the same for `.../pkgs/r`, once per machine |
| `zsh: no matches found: nemo_toolkit[asr]` | zsh (macOS's default shell) treats square brackets as glob pattern characters and tries to expand them before pip ever sees them | Quoted the package spec: `pip install "nemo_toolkit[asr]" --break-system-packages` |
| `ModuleNotFoundError: No module named 'nemo'` | Ran the script from the `(base)` conda environment instead of `(asr)` — a fresh terminal always defaults back to `(base)` | Ran `conda activate asr` before every new terminal session, confirmed by checking the prompt showed `(asr)` |

**What we learned:** Environment setup friction is disproportionately about shell
quirks (zsh globbing) and tooling changes (conda's new ToS gate) rather than the ML
stack itself — worth budgeting real time for this even before touching any ASR code.
Also: conda's per-terminal environment reset is a very easy, very common trap.

---

### Step 2 — Offline Transcription Sanity Check

**What we did:** Ran the `nvidia/parakeet-tdt-1.1b` model via NeMo's
`ASRModel.from_pretrained()` on a self-recorded test clip to confirm the environment
could actually produce a transcript before adding any streaming complexity.

**Errors hit and how we fixed them:**

| Error / Confusion | Cause | Fix |
|---|---|---|
| Wall of NeMo logging (dataloader configs, warnings about `setup_training_data()`) on first run | Normal NeMo behavior — these warnings only matter for training/fine-tuning, not inference | Confirmed this is expected output, not an error; learned to look past it for the actual result |
| Transcript buried in a `Hypothesis(...)` object, hard to read at a glance | `model.transcribe()` returns rich `Hypothesis` objects with internal decoder state (`y_sequence`, `dec_state`, `alignments`, etc.), not a plain string | Accessed the transcript directly via `transcript[0].text` instead of printing the whole object |

**Result:** First successful transcript — the model correctly transcribed a
self-recorded test clip including a proper name, confirming the environment was
correctly set up end-to-end.

**What we learned:** NeMo's verbose logging is a genuine readability hurdle for
newcomers, and the `Hypothesis` object's rich internal state is easy to mistake for
an error message on first encounter. Knowing to isolate `.text` early saves a lot of
squinting later.

---

### Step 3 — Gaming-Vocabulary Evaluation Set + WER Benchmarking

**What we did:** Considered an existing dataset
([Nexdata's gaming speech sample on Kaggle](https://www.kaggle.com/datasets/nexdatafrank/english-gaming-speech-dataset))
but found it was only 11 clips — too small for a reliable WER measurement — so it was
set aside as a bonus/held-out set rather than the primary eval set. Built a
50-phrase, self-recorded gaming-vocabulary eval set instead (callouts, ability names,
tactical commands, slang), matched against a `ground_truth.json`, and ran WER
comparisons across Parakeet, Whisper, and Moonshine.

**Errors hit and how we fixed them:**

| Error | Cause | Fix |
|---|---|---|
| Kaggle dataset only contained 11 `.wav`/`.txt` pairs | It's a preview sample of a larger commercial dataset, not the full corpus | Used it only as a small supplementary/held-out set; built the primary 50-clip eval set by hand instead |
| `FileNotFoundError` / `AudioLoadingError` when computing WER | `ground_truth.json` keys didn't include the folder path prefix (`custom_data/`) that the actual files lived under | Regenerated `ground_truth.json` using a script that auto-lists real files from `os.listdir()`, so keys always match real paths |
| `AttributeError: partially initialized module 'whisper' has no attribute 'load_model'` | Script was named `whisper.py`, which shadowed the actual installed `whisper` package on import (Python checks the current directory before site-packages) | Renamed the script to `run_whisper.py` |

**Result:** A working, repeatable WER pipeline across three models on a
domain-relevant vocabulary set, plus a documented rationale for why a public/
commercial sample dataset alone wasn't sufficient.

**What we learned:** Two non-obvious traps ate more time than the actual ML code:
(1) file-path consistency between a manifest/ground-truth file and the actual
directory structure, and (2) naming a script the same as a package it imports. Both
are "boring" bugs, but they're exactly the kind of thing that silently derails a
project if you don't recognize the pattern quickly. On the data side: a tiny sample
dataset can still be useful as a secondary check, but shouldn't be mistaken for a
sufficient primary benchmark.

---

### Step 4 — Converting to Streaming (Cache-Aware) Inference

**What we did:** Cloned NeMo's `asr_cache_aware_streaming` example, built a JSONL
manifest from the eval clips, and ran `speech_to_text_cache_aware_streaming_infer.py`
against the `parakeet-tdt_ctc-110m` streaming-tuned checkpoint with explicit
`chunk_size`, `shift_size`, and `left_chunks` overrides.

**Errors hit and how we fixed them:**

| Error | Cause | Fix |
|---|---|---|
| `unrecognized arguments: --asr_model=... --manifest_file=...` | The script uses **Hydra** for config, not argparse — Hydra config overrides are plain `key=value` pairs with no leading dashes; `--`-style flags are reserved for Hydra's own machinery | Ran `python script.py --help` (on its own, with no other args) to see the real Hydra config schema, then used the correct field names with no dashes |
| `--help` itself threw "unrecognized arguments" | Ran `--help` combined with other flags on the same command line — Hydra's parser validates all arguments before `--help` gets to short-circuit and print | Ran `--help` completely alone, with nothing else on the line |
| Wrong config field guessed (`manifest_file` instead of the real `dataset_manifest`) | Generic examples don't always match the exact field names in a given NeMo release | Read the actual `--help` output for this specific installed version and used the confirmed real name, `dataset_manifest` |
| `soundfile.LibsndfileError: Format not recognised` | Recorded clips were actually AAC/`.m4a` audio internally, just renamed with a `.wav` extension (confirmed via `file custom_data/clip_001.wav`, which reported "Apple iTunes ALAC/AAC-LC") | Re-encoded every clip to genuine 16kHz mono PCM WAV using `ffmpeg`, output to a new `custom_data_fixed/` folder |
| `FileNotFoundError: '/full/path/to/manifest.json'` | Copy-pasted a literal placeholder path from the doc instead of substituting the real path | Used `pwd` in the manifest's actual folder to get the real absolute path, then substituted it |
| `WER% of streaming mode: inf` | Manifest's `text` field was intentionally left blank (`""`) when first built, since ground truth wasn't required just to prove streaming worked; WER against an empty string is undefined | Updated the manifest-generation script to pull real ground truth from `ground_truth.json` by matching filenames, producing a populated `text` field |
| `WER% of streaming mode: 70.28` (high, but not a bug) | Compounding effects: (1) using the smaller, faster `110m` checkpoint rather than the `1.1b` model, (2) an aggressive `chunk_size=16` giving the model very little context per step, (3) raw WER penalizing casing/punctuation mismatches (e.g. `"Push mid. Now."` vs. ground truth `"push mid now"`) unless both sides are normalized first | Interpreted as a real, informative result rather than an error; planned a follow-up comparison using the `1.1b` model and a larger chunk size to quantify how much of the gap closes |

**Result:** A fully working, confirmed cache-aware streaming pipeline: 50 clips
streamed through the model in ~12 seconds, producing real (if imperfect) partial
transcripts, with the `CacheAwareStreamingConfig(...)` log line confirming the true
streaming code path — not the offline path from Step 2 — was actually exercised.

**What we learned:** This was the highest-friction, highest-payoff phase. The
biggest recurring theme was **configuration system mismatch** — assuming a
standard argparse CLI when the actual tool used Hydra, which changes both the syntax
(no dashes) and the discovery method (`--help` must be run alone). The audio format
mismatch (`AAC` mislabeled as `.wav`) was a reminder that "the file extension says
X" and "the file actually is X" are different claims, and `file <filename>` is a
fast, cheap way to check which one is true. On the modeling side: a 70% WER on a
small, fast, aggressively-chunked model isn't a failed experiment — it's a
quantified data point in exactly the latency/accuracy tradeoff space this project
exists to explore, and it's more interview-useful *with* the failure modes shown than
without them.

---

### Step 5 — Voice Activity Detection (VAD) — *Up Next*

**Status:** Not yet implemented in this project as of this README.

**What it will involve:** Installing `silero-vad`, running it as a lightweight gate
in front of the streaming ASR model so audio is only forwarded to the (expensive)
ASR model when speech is actually detected, and tuning the speech-probability
threshold against real clips. This sets up the turn-taking and barge-in logic in
Step 6.

---

## 4. Key Takeaways So Far

- **Tooling friction dominates early-stage time cost.** Across five steps, more debugging time went into shell quirks, config-system mismatches (argparse vs. Hydra), and file-format mismatches than into anything resembling "the hard ML part." This is itself worth stating explicitly in an interview — knowing how to diagnose *category* of failure (environment vs. data vs. model) quickly is a real skill.
- **Verbose logging isn't the same as an error.** NeMo in particular prints a large amount of informational and warning-level output on every run; learning to scan past it for the actual result (or the actual traceback) is a distinct skill from reading a clean, well-behaved CLI tool.
- **Small model + small chunk size = a real, quantifiable accuracy cost.** The 70.28% streaming WER on the `110m` model with `chunk_size=16` isn't a failure state — it's a first concrete data point in the latency-vs-accuracy curve this project is built to characterize, and the natural next step is to vary those two knobs and observe how the number moves.
- **File format claims need verification, not assumption.** A `.wav` extension does not guarantee WAV-encoded audio; `file <filename>` is a five-second check that would have caught the AAC mismatch immediately instead of surfacing as a cryptic `libsndfile` error two layers downstream.
- **Domain-specific evaluation data matters more than dataset size for relevance, but size matters for reliability.** An 11-clip commercial sample was gaming-domain-relevant but too small to trust as a primary metric; a self-built 50-clip set traded some domain diversity for statistical reliability and full control over the exact phrases tested.

---

## 5. Related Research Papers

Papers and technical reports directly relevant to the models, architectures, and
techniques used or referenced in this project:

- **Conformer: Convolution-augmented Transformer for Speech Recognition** (Gulati et al., 2020) — the encoder architecture family that FastConformer builds on, combining convolution and self-attention for speech modeling.
- **FastConformer / "Stateful FastConformer"** (NVIDIA, 2023–2024 technical reports) — the downsampled, compute-efficient Conformer variant used as Parakeet's encoder backbone, including the cache-aware streaming formulation used in this project.
- **RNN-Transducer: Sequence Transduction with Recurrent Neural Networks** (Graves, 2012) — the original RNN-T formulation underlying transducer-based streaming ASR decoders.
- **TDT: Efficient Sequence Transduction by Jointly Predicting Tokens and Durations** (Xu et al., NVIDIA, 2023) — the Token-and-Duration Transducer decoding approach used by the Parakeet-TDT models in this project, which reduces decoding steps by predicting skip-durations jointly with tokens.
- **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision** (Radford et al., OpenAI, 2022) — the general-purpose ASR model used as a comparison baseline in the Step 3 WER benchmarking.
- **Moonshine** (Useful Sensors, 2024) — a small, low-latency-optimized ASR model family used as a second comparison baseline, relevant given this project's own low-latency/edge-deployment goals.
- **Silero VAD** (Silero Team) — the lightweight voice activity detection model planned for Step 5, used broadly in production streaming-speech systems for its speed/accuracy tradeoff on CPU.
- **Mimi: a streaming neural audio codec** (Kyutai / Moshi team, 2024) — relevant to a stretch goal of this project (feeding codec-compressed audio into the ASR pipeline) and directly related to Frisson Labs' own published work on full-duplex voice architectures.
- **Moshi: a speech-text foundation model for real-time dialogue** (Kyutai, 2024) — the full-duplex conversational architecture that motivates this project's eventual barge-in/turn-taking design in Steps 6+.
