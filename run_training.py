import os
import sys

# Add current dir (project root) and subfolders to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: Confirm path
# print("PYTHONPATH includes:", sys.path)

from training.train import main

if __name__ == "__main__":
    print("Running MARL training from run_training.py")
    main()