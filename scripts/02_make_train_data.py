import json
from pathlib import Path

# Find project root (EdgeIOT/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def list_folders(base_dir: Path):
    return sorted([f for f in base_dir.iterdir() if f.is_dir()])


def choose_folder(folders):
    print("\nAvailable output runs:\n")

    for i, f in enumerate(folders):
        print(f"[{i}] {f.name}")

    idx = input("\nSelect index: ").strip()

    if not idx.isdigit():
        raise ValueError("Index must be a number")

    idx = int(idx)

    if idx < 0 or idx >= len(folders):
        raise IndexError("Index out of range")

    return folders[idx]


def find_semantic_file(folder: Path):
    # Some runs have nested structure, so search recursively
    matches = list(folder.rglob("semantic_outputs.json"))

    if not matches:
        raise FileNotFoundError(f"No semantic_outputs.json found in {folder}")

    return matches[0]


def convert(folder: Path):
    src = find_semantic_file(folder)
    out = folder / "train_data.jsonl"

    print(f"\nUsing: {src}")

    with src.open("r") as f:
        records = json.load(f)

    with out.open("w") as f:
        for r in records:
            prompt = " ".join(r.get("semantic_tokens", []))
            response = r.get("llm_reasoning", "")

            json.dump(
                {"instruction": prompt, "output": response},
                f
            )
            f.write("\n")

    print(f"\nSaved: {out}")


def main():
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Missing outputs directory: {OUTPUT_DIR}")

    folders = list_folders(OUTPUT_DIR)

    if not folders:
        raise RuntimeError("No run folders found")

    folder = choose_folder(folders)

    print(f"\nSelected: {folder.name}")

    convert(folder)


if __name__ == "__main__":
    main()