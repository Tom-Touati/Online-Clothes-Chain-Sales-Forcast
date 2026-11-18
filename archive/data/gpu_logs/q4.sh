cd ~/m5 && source ~/venv/bin/activate
python eda/57_embed_nn.py seeds=3 threads=2 inter=0 name=1 tag=_N13 > full_N13.log 2>&1
python eda/57_embed_nn.py seeds=3 seed0=10 threads=2 inter=0 name=1 tag=_N13s10 > full_N13s10.log 2>&1
