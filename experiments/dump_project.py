# scripts/dump_project.py
# Run this from your project root: python scripts/dump_project.py
# It will create mrag_project_dump.txt (one single file with everything)

import os
from pathlib import Path

def should_include_file(file_path: str) -> bool:
    """Only include text/source files, skip binaries and junk."""
    ext = Path(file_path).suffix.lower()
    text_exts = {
        '.py', '.md', '.toml', '.yaml', '.yml', '.txt',
        '.json', '.env', '.example', '.gitignore', '.pre-commit-config.yaml'
    }
    return ext in text_exts

def should_skip_dir(dir_name: str) -> bool:
    skip = {
        '__pycache__',
        'models-cache',
        '.git',
        'venv',
        'env',
        '.venv',
        'node_modules',
        '__pycache__',
        '.pytest_cache',
        '.ruff_cache',
        'dist',
        'build'
    }
    return dir_name in skip or dir_name.startswith('.')

def dump_project(output_file: str = "mrag_project_dump.txt"):
    root = os.getcwd()
    total_files = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# ===============================================\n")
        f.write("# M-RAG FULL PROJECT DUMP\n")
        f.write("# Generated automatically - ready to share\n")
        f.write("# ===============================================\n\n")
        
        for dirpath, dirnames, filenames in os.walk(root):
            # Remove directories we want to skip
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
            
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, root)
                
                if should_include_file(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as src:
                            content = src.read()
                        
                        f.write(f"\n\n=== FILE: {rel_path} ===\n")
                        f.write("=" * 60 + "\n")
                        f.write(content)
                        f.write("\n" + "=" * 60 + "\n")
                        total_files += 1
                        print(f"✓ {rel_path}")
                    except Exception as e:
                        print(f"⚠️  Skipped (encoding issue): {rel_path}")
    
    print(f"\n✅ DONE! Created {output_file} with {total_files} files.")
    print("   Just share this one file with me or Claude.")

if __name__ == "__main__":
    dump_project()