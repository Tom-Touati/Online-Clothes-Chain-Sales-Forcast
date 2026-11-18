cd ~/m5 && source ~/venv/bin/activate
python eda/57_embed_nn.py seeds=3 threads=2 inter=0 seq=1 origins=2026-06 tag=_finalS > finalS.log 2>&1
python eda/57_embed_nn.py seeds=3 seed0=10 threads=2 inter=0 seq=1 origins=2026-06 tag=_finalS_s10 > finalS_s10.log 2>&1
