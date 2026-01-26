import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from src.utils import load_config
config = load_config()
from src.dataset_creation import load_initial_dataset
initial_data = load_initial_dataset(Path(config['dataset']['raw_path']) / 'initial_dataset.json')
print(f"Loaded {len(initial_data)} initial examples")