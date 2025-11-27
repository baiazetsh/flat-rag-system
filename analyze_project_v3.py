# analyze_project v6 — STRICT CLEAN CODE ONLY
import os
import json

# -------- DIRECTORIES TO EXCLUDE COMPLETELY --------
EXCLUDE_DIRS = {
    "__pycache__", ".git", "venv", "env", ".idea", ".vscode",
    "node_modules", "redis", "postgres", "backup", "cache",
    "htmlcov", "dist", "build", "coverage", ".pytest_cache",
    "chroma_db", "qdrant", "data", "database", "db", "storage",
    "vectorstore", "datasets", "logs", "app/logs",
}

# -------- EXTENSIONS TO INCLUDE (ONLY CODE!) --------
INCLUDE_EXTS = {
    ".py", ".env", ".yaml",
}

# -------- SPECIAL FILENAMES (ONLY DEVOPS CODE) --------
INCLUDE_FILENAMES = {
    "Dockerfile", "dockerfile",
    "requirements.txt",
    "compose.yml", "compose.yaml",
    "Makefile",
}

# -------- HARD LIMIT: NO FILE > 200 KB --------
MAX_BYTES = 200 * 1024


def collect_files(root: str):
    dataset = []

    for dirpath, dirnames, filenames in os.walk(root):

        # remove excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            # include logic
            include = (
                ext in INCLUDE_EXTS or
                filename in INCLUDE_FILENAMES
            )

            if not include:
                continue

            # size limit
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
            except:
                continue

            # read file
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                content = f"[READ ERROR: {e}]"

            dataset.append({
                "path": path,
                "name": filename,
                "extension": ext,
                "content": content,
            })

    # sort deterministically
    dataset.sort(key=lambda x: x["path"])
    return dataset


def save_as_json(dataset, out_file="project_fulltext.json"):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Saved {len(dataset)} files → {out_file}")


if __name__ == "__main__":
    data = collect_files(".")
    save_as_json(data)
