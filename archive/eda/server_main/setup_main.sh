set -e
cd ~/edikted_main
python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
venv/bin/pip install -q pandas pyarrow numpy scipy scikit-learn lightgbm neuralforecast open_clip_torch sentence-transformers pytrends pillow requests
venv/bin/python -c "import torch, neuralforecast, open_clip, sentence_transformers; print('ok', torch.__version__, torch.cuda.is_available())"
