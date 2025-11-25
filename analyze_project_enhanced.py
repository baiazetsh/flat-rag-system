#!/usr/bin/env python3
"""
Project analyzer - collects source files with smart filtering and metadata.
"""
import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime


# --- Исключаем мусор и тяжёлые директории ---
EXCLUDE_DIRS = {
    "__pycache__", ".git", "venv", "env", ".idea",
    ".vscode", "migrations", "node_modules",
    "db", "database", "data", "qdrant", "chroma",
    "redis", "postgres", "backup", "cache", "staticfiles",
    "dist", "build", ".next", ".nuxt", "coverage",
    ".pytest_cache", ".mypy_cache", "htmlcov"
}

# --- Исключаем отдельные файлы ---
EXCLUDE_FILES = {
    ".DS_Store", "Thumbs.db", ".gitignore",
    "package-lock.json", "yarn.lock", "poetry.lock",
}

# --- Включаем расширения ---
INCLUDE_EXTS = (
    ".py", ".html", ".css", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".yml", ".yaml",
    ".ini", ".cfg", ".toml",
    ".env", ".env.example",
    ".txt", ".md", ".rst",
    ".sql", ".sh", ".bash",
)

INCLUDE_FILES = {
    "Dockerfile", "dockerfile",
    "requirements.txt", "setup.py", "pyproject.toml",
    "Makefile", ".gitignore"
}


def get_file_hash(content: str) -> str:
    """Возвращает SHA256 хеш содержимого файла."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def get_file_stats(filepath: str) -> dict:
    """Собирает статистику файла."""
    stat = os.stat(filepath)
    return {
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }


def should_include_file(filename: str, filepath: str, max_size_mb: float = 1.0) -> tuple[bool, str]:
    """
    Проверяет, нужно ли включать файл.
    
    Returns:
        (should_include, reason)
    """
    # Проверка размера
    size_bytes = os.path.getsize(filepath)
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"too large ({size_bytes / 1024 / 1024:.2f}MB)"
    
    # Проверка исключений
    if filename in EXCLUDE_FILES:
        return False, "excluded file"
    
    # Проверка расширения
    if filename.endswith(INCLUDE_EXTS) or filename in INCLUDE_FILES:
        return True, "match"
    
    return False, "unknown extension"


def collect_files(
    root: str,
    *,
    max_size_mb: float = 1.0,
    include_stats: bool = True,
    include_hash: bool = False,
    relative_paths: bool = True,
    verbose: bool = False
) -> list[dict[str, str]]:
    """
    Собирает файлы проекта с фильтрацией и метаданными.
    
    Args:
        root: Корневая директория проекта
        max_size_mb: Максимальный размер файла в МБ
        include_stats: Включить статистику файлов
        include_hash: Включить хеш содержимого
        relative_paths: Использовать относительные пути
        verbose: Выводить подробную информацию
    """
    dataset = []
    skipped = []
    root_path = Path(root).resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        # Фильтруем каталоги
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            
            # Проверяем, нужно ли включать файл
            should_include, reason = should_include_file(filename, filepath, max_size_mb)
            
            if not should_include:
                if verbose:
                    skipped.append((filepath, reason))
                continue

            # Читаем содержимое
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                content = f"⚠️ [READ ERROR: {e}]"
                if verbose:
                    print(f"⚠️  {filepath}: {e}")

            # Формируем запись
            file_path = Path(filepath).resolve()
            try:
                rel_path = file_path.relative_to(root_path) if relative_paths else file_path
                path_str = str(rel_path)
            except ValueError:
                # Если файл вне root_path, используем абсолютный путь
                path_str = str(file_path)
            
            record = {
                "path": path_str,
                "name": filename,
                "extension": file_path.suffix,
                "content": content,
            }

            # Добавляем опциональные метаданные
            if include_stats:
                try:
                    record["stats"] = get_file_stats(filepath)
                except Exception as e:
                    record["stats"] = {"error": str(e)}

            if include_hash:
                record["hash"] = get_file_hash(content)

            dataset.append(record)

    if verbose:
        print(f"\n📊 Статистика:")
        print(f"   ✅ Собрано: {len(dataset)} файлов")
        print(f"   ⏭️  Пропущено: {len(skipped)} файлов")
        if skipped:
            print(f"\n📋 Пропущенные файлы (первые 10):")
            for path, reason in skipped[:10]:
                print(f"   • {path}: {reason}")

    return dataset


def save_as_json(
    dataset: list[dict],
    out_file: str = "project_fulltext.json",
    pretty: bool = False
):
    """Сохраняет датасет в JSON."""
    with open(out_file, "wt", encoding="utf-8") as f:
        if pretty:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        else:
            json.dump(dataset, f, ensure_ascii=False, separators=(",", ":"))
    
    file_size = os.path.getsize(out_file) / 1024 / 1024
    print(f"✅ Сохранено {len(dataset)} файлов → {out_file} ({file_size:.2f}MB)")


def save_as_markdown(
    dataset: list[dict],
    out_file: str = "project_fulltext.md"
):
    """Сохраняет в Markdown для удобного просмотра."""
    with open(out_file, "wt", encoding="utf-8") as f:
        f.write(f"# Project Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n")
        f.write(f"**Total files:** {len(dataset)}\n\n")
        f.write("---\n\n")

        for item in dataset:
            f.write(f"## {item['path']}\n\n")
            
            if "stats" in item:
                stats = item["stats"]
                f.write(f"- **Size:** {stats.get('size_bytes', 0) / 1024:.2f}KB\n")
                f.write(f"- **Modified:** {stats.get('modified', 'N/A')}\n")
            
            if "hash" in item:
                f.write(f"- **Hash:** `{item['hash']}`\n")
            
            f.write(f"\n```{item['extension'][1:]}\n")
            f.write(item['content'])
            f.write(f"\n```\n\n---\n\n")
    
    print(f"✅ Markdown сохранён → {out_file}")


def get_project_summary(dataset: list[dict]) -> dict:
    """Генерирует краткую сводку проекта."""
    extensions = {}
    total_lines = 0
    total_size = 0

    for item in dataset:
        ext = item['extension'] or 'no-ext'
        extensions[ext] = extensions.get(ext, 0) + 1
        
        lines = item['content'].count('\n')
        total_lines += lines
        
        if 'stats' in item:
            total_size += item['stats'].get('size_bytes', 0)

    return {
        "total_files": len(dataset),
        "total_lines": total_lines,
        "total_size_mb": total_size / 1024 / 1024,
        "files_by_extension": dict(sorted(extensions.items(), key=lambda x: x[1], reverse=True)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze project structure and collect source files"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to analyze (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="project_fulltext.json",
        help="Output JSON file (default: project_fulltext.json)"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also save as Markdown"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=1.0,
        help="Maximum file size in MB (default: 1.0)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't include file statistics"
    )
    parser.add_argument(
        "--with-hash",
        action="store_true",
        help="Include content hashes"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print project summary"
    )

    args = parser.parse_args()

    print(f"🔍 Анализируем проект: {args.root}\n")

    # Собираем файлы
    dataset = collect_files(
        args.root,
        max_size_mb=args.max_size,
        include_stats=not args.no_stats,
        include_hash=args.with_hash,
        verbose=args.verbose
    )

    # Сохраняем JSON
    save_as_json(dataset, args.output, pretty=args.pretty)

    # Сохраняем Markdown (опционально)
    if args.markdown:
        md_file = args.output.replace('.json', '.md')
        save_as_markdown(dataset, md_file)

    # Выводим сводку (опционально)
    if args.summary or args.verbose:
        summary = get_project_summary(dataset)
        print(f"\n📈 Сводка проекта:")
        print(f"   Всего файлов: {summary['total_files']}")
        print(f"   Всего строк: {summary['total_lines']:,}")
        print(f"   Общий размер: {summary['total_size_mb']:.2f}MB")
        print(f"\n   Файлы по типам:")
        for ext, count in list(summary['files_by_extension'].items())[:10]:
            print(f"     {ext:15} {count:>4} файлов")


if __name__ == "__main__":
    main()
"""
# Базовое использование
python analyze_project.py

# С опциями
python analyze_project.py /path/to/project -o output.json --verbose --summary

# Markdown экспорт
python analyze_project.py --markdown --pretty

# Контроль размера
python analyze_project.py --max-size 2.0 --with-hash
"""