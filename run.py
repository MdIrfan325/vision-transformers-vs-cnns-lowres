import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the training script
from train import main

if __name__ == "__main__":
    main() 