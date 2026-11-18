cd ~/m5 && source ~/venv/bin/activate
O=2025-02,2025-03,2025-04,2025-05
for cfg in "A " "B inter=0" "C inter=0 item=0" "D epochs=15" "E pen=10"; do
  set -- $cfg; tag=$1; shift
  python eda/57_embed_nn.py seeds=2 threads=2 origins=$O tag=_s$tag "$@" > sweep_$tag.log 2>&1
done
echo done > sweep_done
