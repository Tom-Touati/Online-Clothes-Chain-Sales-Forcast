python3 -m venv venv_pfn
venv_pfn/bin/pip install -q --upgrade pip
venv_pfn/bin/pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
venv_pfn/bin/pip install -q tabpfn==10.0.0 pandas pyarrow
venv_pfn/bin/python -c "import torch, tabpfn; print(\"OK\", torch.__version__, torch.cuda.is_available(), tabpfn.__version__)"
