import os
import argparse
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def extract_features(input_dir, output_dir, device):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

    os.makedirs(output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} audio files in {input_dir}")

    for idx, file in enumerate(wav_files):
        file_path = os.path.join(input_dir, file)
        waveform, sr = torchaudio.load(file_path)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model(**inputs).last_hidden_state.cpu()

        torch.save(features, os.path.join(output_dir, file.replace(".wav", ".pt")))
        print(f"[{idx+1}/{len(wav_files)}] Extracted features for {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to folder containing wav files")
    parser.add_argument("n_threads", type=int, help="Number of threads (not used in this example)")
    parser.add_argument("gpu_index", type=int, help="GPU index or -1 for CPU")
    parser.add_argument("exp_dir", type=str, help="Experiment directory (logs/speaker_name)")
    parser.add_argument("version", type=str, help="Version (e.g., v2)")
    parser.add_argument("if_f0", type=str, help="Whether to use f0 features")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_index}" if args.gpu_index >= 0 and torch.cuda.is_available() else "cpu")

    # تحويل المسار إلى Path ثم استخراج المجلد الأب
    exp_dir_path = Path(args.exp_dir)
    output_dir = exp_dir_path / "3_feature256"

    extract_features(args.input_dir, output_dir, device)
