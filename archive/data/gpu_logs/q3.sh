cd ~/m5 && source ~/venv/bin/activate
while pgrep -f "tag=_T13" > /dev/null; do sleep 15; done
python eda/57_embed_nn.py seeds=3 seed0=10 threads=2 inter=0 origins=2026-06 tag=_final_s10 > final_s10.log 2>&1
