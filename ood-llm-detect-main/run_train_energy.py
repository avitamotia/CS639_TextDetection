"""
Launcher for energy-mode training on the Deepfake dataset.
Handles weight download (replaces wget) and sets Windows/single-GPU-friendly defaults.

Usage:
    python run_train_energy.py
    python run_train_energy.py --no_pretrain   # train from scratch without downloading weights
    python run_train_energy.py --epochs 10     # quick test run
"""
import os
import sys
import argparse
import subprocess
import urllib.request

CKPT_DIR = "./ckpt"
DEEPFAKE_PTH = os.path.join(CKPT_DIR, "Deepfake_best.pth")
DEEPFAKE_PTH_URL = "https://huggingface.co/heyongxin233/DeTeCtive/resolve/main/Deepfake_best.pth"


def download_weights(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"Checkpoint already exists: {dest}")
        return
    print(f"Downloading pretrained weights to {dest} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 1024
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
        print(f"\nDownload complete: {dest}")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Falling back to training from scratch (--resum False).")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_pretrain", action="store_true", help="Skip weight download and train from scratch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256, help="Max token length; lower = less VRAM")
    args = parser.parse_args()

    use_pretrain = not args.no_pretrain
    pth_path = ""

    if use_pretrain:
        ok = download_weights(DEEPFAKE_PTH_URL, DEEPFAKE_PTH)
        if ok:
            pth_path = DEEPFAKE_PTH
        else:
            use_pretrain = False

    cmd = [
        sys.executable, "train_classifier_energy.py",
        "--device_num", "1",
        "--per_gpu_batch_size", str(args.batch_size),
        "--per_gpu_eval_batch_size", str(args.eval_batch_size),
        "--max_length", str(args.max_length),
        "--total_epoch", str(args.epochs),
        "--lr", "2e-5",
        "--warmup_steps", "1000",
        "--method", "energy",
        "--classifier_dim", "7",
        "--model_name", "princeton-nlp/unsup-simcse-roberta-base",
        "--dataset", "deepfake",
        "--path", "data/Deepfake/cross_domains_cross_models",
        "--name", "deepfake-roberta-base",
        "--freeze_embedding_layer",
        "--database_name", "train",
        "--test_dataset_name", "test",
        "--num_workers", "0",
        "--precision", "16-mixed",
    ]

    # argparse type=bool treats any non-empty string as True, so only pass --resum when
    # we actually have a checkpoint to load
    if use_pretrain and pth_path:
        cmd += ["--resum", "True", "--pth_path", pth_path]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
