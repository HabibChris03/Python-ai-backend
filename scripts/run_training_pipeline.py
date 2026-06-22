"""
End-to-end: generate images from JSON labels, then fine-tune CLIP.

    cd ai_backend
    python scripts/run_training_pipeline.py
"""
import subprocess
import sys


def run(command):
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    run([sys.executable, "scripts/prepare_training_images.py", "--data-root", "data/training_docs"])
    run(
        [
            sys.executable,
            "scripts/train_classifier.py",
            "--data-root",
            "data/training_docs",
            "--output-dir",
            "models/finetuned_clip_v2",
            "--batch-size",
            "8",
            "--epochs",
            "10",
        ]
    )
    print("\nTraining pipeline complete. Restart the AI backend to load the new model.")
