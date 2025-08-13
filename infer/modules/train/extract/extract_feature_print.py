import os
import torch
import torchaudio
import argparse
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def extract_features(input_dir, output_dir, device="cpu"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # تحميل نموذج HuBERT Base
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
    model = Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")
    model.to(device)
    model.eval()

    wav_files = sorted(input_dir.glob("*.wav"))
    total = len(wav_files)
    print(f"Found {total} wav files.")

    for idx, wav_path in enumerate(wav_files, 1):
        print(f"[{idx}/{total}] Processing {wav_path.name}...")
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        
        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        input_values = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            features = model(input_values).last_hidden_state.squeeze(0)  # shape: (seq_len, 768)
        
        save_path = output_dir / (wav_path.stem + ".pt")
        torch.save(features.cpu(), save_path)

    print("All features saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to 16k wav files")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to save features (default: 3_feature256)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir.parent, "3_feature256")

    extract_features(args.input_dir, args.output_dir, args.device)
