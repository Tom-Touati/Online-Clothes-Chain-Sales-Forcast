cd ~/m5 && source ~/venv/bin/activate
while pgrep -f "tag=_B13" > /dev/null; do sleep 20; done
python eda/57_embed_nn.py seeds=3 threads=2 inter=0 camp=2 tag=_B13c2 > full_B13c2.log 2>&1
python eda/57_embed_nn.py seeds=3 threads=2 inter=0 camp=1 tag=_B13c1 > full_B13c1.log 2>&1
