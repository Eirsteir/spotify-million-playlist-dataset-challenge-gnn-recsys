

## Development Environment
* Python Anaconda v4.4.10  
* Tensorflow v1.5.0  
* CUDA Toolkit v9.0 and cuDNN v7.0  
* GPU: 4 Nvidia GTX 1080Ti  

## Requirements

Base packages are required to be installed before PyTorch Geometric etc. Run this for a correct installation: 

For CPU (pip):
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install --no-index --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install --no-index --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install --no-index --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install --no-index --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.11.0+cpu.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

## Library
* PyTorch Geometric

## Dataset
Spotify has produced the MPD(Million Playlist Dataset) which contains a million user-curated playlists. 
Each playlist in the MPD contains a playlist title, a list of tracks(with metadata), and other miscellaneous information. 

## Preprocess The Data