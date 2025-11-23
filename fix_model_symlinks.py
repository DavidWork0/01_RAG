import os
import shutil
from pathlib import Path

def resolve_symlinks_to_files(start_path):
    """
    Walks through a directory, finds symlinks, and replaces them with the actual files they point to.
    """
    start_path = Path(start_path)
    if not start_path.exists():
        print(f"Path not found: {start_path}")
        return

    print(f"Scanning {start_path} for symlinks...")
    count = 0
    
    for path in start_path.rglob('*'):
        if path.is_symlink():
            try:
                # Get the target of the symlink
                target = path.resolve()
                
                if not target.exists():
                    print(f"⚠️  Broken link: {path} -> {target}")
                    continue
                
                # It's a valid symlink. We want to replace the link with the file.
                print(f"Fixing: {path.name}")
                
                # 1. Unlink (remove) the symlink
                path.unlink()
                
                # 2. Copy the actual file to this location
                if target.is_dir():
                    shutil.copytree(target, path)
                else:
                    shutil.copy2(target, path)
                
                count += 1
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")

    print(f"\nDone! Replaced {count} symlinks with actual files.")

if __name__ == "__main__":
    # Path to your huggingface models
    models_dir = os.path.join(os.path.dirname(__file__), "models", "huggingface")
    resolve_symlinks_to_files(models_dir)
