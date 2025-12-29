
import os
import sys
import subprocess
from pathlib import Path

def download_file(url, output_path):
    """Download a file from a URL to a specific output path."""
    import gdown
    
    # Create parent directories if they don't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"File already exists at {output_path}, skipping download.")
        return

    print(f"Downloading to {output_path}...")
    try:
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        print(f"Successfully downloaded {output_path}")
    except Exception as e:
        print(f"Failed to download {output_path}: {e}")

def main():
    # Model 1: best.pt
    best_pt_url = "https://drive.google.com/file/d/11-tN3GAnxLvrSUNimJzKSldJzzXosE9d/view?usp=drive_link"
    best_pt_path = "models/best.pt"
    download_file(best_pt_url, best_pt_path)

    # Model 2: reid/net.pth
    net_pth_url = "https://drive.google.com/file/d/1vJwbdxL8bq3XH7SW5k52YFEk38O7frdw/view?usp=sharing"
    net_pth_path = "models/reid/net.pth"
    download_file(net_pth_url, net_pth_path)

    # Model 3: reid/opts.yaml
    opts_yaml_url = "https://drive.google.com/file/d/1m0-YfUm3OjgdhjbrOz_Wr5nOQT7zwB8-/view?usp=drive_link"
    opts_yaml_path = "models/reid/opts.yaml"
    download_file(opts_yaml_url, opts_yaml_path)

if __name__ == "__main__":
    main()
