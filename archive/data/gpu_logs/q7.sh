cd ~/m5 && source ~/venv/bin/activate
python eda/57_embed_nn.py seeds=3 threads=2 inter=0 seq=1 sl=52 tag=_L13 > full_L13.log 2>&1
python eda/57_embed_nn.py seeds=3 seed0=10 threads=2 inter=0 seq=1 sl=52 tag=_L13s10 > full_L13s10.log 2>&1
