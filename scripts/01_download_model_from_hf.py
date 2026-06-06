import subprocess
import json
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "languagemodels"


def get_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def search_models(query: str, limit: int = 10):
    cmd = [
        "hf",
        "models",
        "ls",
        "--search",
        query,
        "--limit",
        str(limit),
        "--format",
        "json",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    data = json.loads(result.stdout)
    return [item["id"] for item in data]


def get_selection(models):
    while True:
        user_input = input(
            "\nSelect index OR type full model name (or press Enter to retry): "
        ).strip()

        # empty input → retry safely
        if not user_input:
            print("No input detected. Please try again.")
            continue

        # case 1: index selection
        if user_input.isdigit():
            idx = int(user_input)
            if 0 <= idx < len(models):
                return models[idx]
            print("Invalid index. Try again.")
            continue

        # case 2: direct model name
        if "/" in user_input:
            return user_input

        print("Invalid input. Use index or full model id (org/model).")


def main():
    print("\n=== HF Smart Model Downloader ===\n")

    webbrowser.open("https://huggingface.co/models?sort=downloads")

    query = input("Search models (e.g. llama, phi, mistral): ").strip()
    if not query:
        raise ValueError("Search query cannot be empty")

    models = search_models(query)

    if not models:
        print("No models found.")
        return

    print("\nTop results:\n")
    for i, m in enumerate(models):
        print(f"[{i}] {m}")

    model_id = get_selection(models)
    model_name = get_model_name(model_id)

    target_dir = BASE_DIR / model_name

    print("\nSelected:", model_id)
    print("Saving to:", target_dir, "\n")

    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "hf",
        "download",
        model_id,
        "--local-dir",
        str(target_dir),
    ]

    subprocess.run(cmd, check=True)

    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()