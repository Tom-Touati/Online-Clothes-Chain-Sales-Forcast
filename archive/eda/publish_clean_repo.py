"""D420 (user, 18 Nov: "clean the repo so past attempts dont dirty it. a person should be able to read the current solution cleanly. also there
should be one submission file, the per model submission files can be in a per model dir"): publishes this folder (the lab) as the clean
GitHub repository without moving anything here.

The lab keeps its layout (other sessions run scripts against it). The repository's branch `main` is a second git worktree, _clean_repo/, whose
tree is built from a committed snapshot of the lab (branch `lab`, never pushed):
  1 snapshot   git add -A + commit on `lab` (everything not ignored: no .venv, no Dataset.parquet, no _clean_repo)
  2 map        every lab file -> its place in the clean tree, by the rules in eda/publish_map.py (`place(path)`): the one submission file at
               the root, per-model workbooks under model_submissions/<model>/, the tables the solution uses under data/, documents under
               docs/, the rest under archive/<lab path>. A file is in exactly one place.
  3 load       the mapped files go into the index of _clean_repo by blob id (nothing is copied); everything except archive/ is written to
               disk there, archive/ stays index-only (skip-worktree), so the worktree costs little disk.
  4 keep       files that exist only in the clean tree (README.md, solution/..., the folder READMEs: written by hand there) are left as they
               are; docs/FILE_MAP.tsv (lab path -> repository path, for reading the decision log) is regenerated.
  5 commit     on `main` in _clean_repo; --push sends it (HTTPS with the GitHub CLI login).
Usage, from the lab root:  python eda/publish_clean_repo.py [--no-snapshot] [--commit "message"] [--push] [--dry]
Re-run it whenever the lab changes; hand-written files in _clean_repo/ are edited and committed there (this script stages them too)."""
import subprocess, sys, os, io, importlib.util, datetime
from pathlib import Path
LAB = Path(__file__).resolve().parents[1]; CLEAN = LAB / "_clean_repo"; MAPFILE = "docs/FILE_MAP.tsv"
A = sys.argv[1:]; DRY = "--dry" in A; PUSH = "--push" in A; SNAP = "--no-snapshot" not in A
MSG = A[A.index("--commit") + 1] if "--commit" in A else None
URL = "https://github.com/Tom-Touati/Online-Clothes-Chain-Sales-Forcast.git"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
def git(*a, cwd=LAB, inp=None, check=True):
    r = subprocess.run(["git", *a], cwd=str(cwd), input=inp, capture_output=True)
    if check and r.returncode: raise SystemExit(f"git {' '.join(a[:3])} failed in {cwd}:\n{r.stderr.decode('utf-8', 'replace')[-2000:]}")
    return r.stdout
spec = importlib.util.spec_from_file_location("publish_map", LAB / "eda" / "publish_map.py"); pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
assert CLEAN.exists(), "worktree _clean_repo missing: git worktree add --no-checkout _clean_repo main"
assert git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() == "lab", "the lab root must be on branch lab"
# 1 snapshot of the lab
if SNAP and not DRY:
    git("add", "-A")
    if git("status", "--porcelain").strip():
        git("commit", "-q", "-m", "Lab snapshot " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + TRAILER)
snap = git("rev-parse", "lab").decode().strip()
# 2 map every snapshot file
ents = [e for e in git("ls-tree", "-r", "-z", "lab").decode("utf-8").split("\0") if e]
if hasattr(pm, "needed_data"): pm.NEEDED_DATA = pm.needed_data({e.split("\t", 1)[1] for e in ents})
mech, dropped, seen = {}, [], {}
for e in ents:
    meta, path = e.split("\t", 1); mode, typ, sha = meta.split()
    dst = pm.place(path)
    if dst is None: dropped.append(path); continue
    assert dst not in seen, f"two lab files map to {dst}: {seen[dst]} and {path}"
    tr = pm.transform(dst, git("cat-file", "blob", sha).decode("utf-8", "replace")) if hasattr(pm, "transform") and dst.endswith((".py", ".ps1", ".html")) else None
    if tr is not None and not DRY: sha = git("hash-object", "-w", "--stdin", inp=tr.encode("utf-8")).decode().strip()
    seen[dst] = path; mech[dst] = (mode, sha, path)
missing = [p for p in getattr(pm, "REQUIRED", []) if p not in {v[2] for v in mech.values()}]
assert not missing, f"publish_map names lab files that are not in the snapshot: {missing[:10]}"
top = {}
for d in mech: top[d.split("/")[0] if "/" in d else "(root)"] = top.get(d.split("/")[0] if "/" in d else "(root)", 0) + 1
print(f"snapshot {snap[:8]}: {len(ents)} lab files -> {len(mech)} published, {len(dropped)} not published {dropped[:6]}"); print("by top-level folder:", dict(sorted(top.items())))
if DRY: sys.exit()
# 3 load into the clean worktree's index; hand-written files (in the index, not produced by the previous mapping) are kept
prev = set()
old_map = git("show", f"HEAD:{MAPFILE}", cwd=CLEAN, check=False).decode("utf-8", "replace")
first = not old_map.strip()
if not first: prev = {l.split("\t")[1] for l in old_map.splitlines()[1:] if "\t" in l}
cur = [e for e in git("ls-files", "-s", "-z", cwd=CLEAN).decode("utf-8").split("\0") if e]
hand = [] if first else [e for e in cur if e.split("\t", 1)[1] not in prev and e.split("\t", 1)[1] != MAPFILE]
git("read-tree", "--empty", cwd=CLEAN)
info = "".join(f"{m} {s}\t{d}\0" for d, (m, s, _) in mech.items()) + "".join(f"{e.split()[0]} {e.split()[1]}\t{e.split(chr(9), 1)[1]}\0" for e in hand)
git("update-index", "-z", "--index-info", cwd=CLEAN, inp=info.encode("utf-8"))
arch = [d for d in mech if d.startswith("archive/")]; live = [d for d in mech if not d.startswith("archive/")]
if arch: git("update-index", "--skip-worktree", "-z", "--stdin", cwd=CLEAN, inp="\0".join(arch).encode("utf-8"))
if live: git("checkout-index", "-f", "-z", "--stdin", cwd=CLEAN, inp="\0".join(live).encode("utf-8"))
for d in sorted(prev - set(mech)):                      # files of the previous mapping that are no longer published
    f = CLEAN / d
    if f.is_file(): f.unlink()
# 4 the file map
(CLEAN / "docs").mkdir(exist_ok=True)
with io.open(CLEAN / MAPFILE, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("lab_path\trepository_path\n"); [fh.write(f"{src}\t{d}\n") for d, (_, _, src) in sorted(mech.items(), key=lambda kv: kv[1][2])]
git("add", "--", MAPFILE, cwd=CLEAN)
handroots = [p for p in getattr(pm, "HAND_ROOTS", []) if (CLEAN / p).exists()]
if handroots: git("add", "-A", "--", *handroots, cwd=CLEAN)
st = git("status", "--porcelain", cwd=CLEAN).decode("utf-8", "replace").splitlines()
print(f"clean worktree: {len(live)} files on disk, {len(arch)} archive files index-only, {len(hand)} hand-written kept; {len(st)} changes staged/pending")
# 5 commit and push
if MSG:
    if st: git("commit", "-q", "-m", MSG + TRAILER, cwd=CLEAN)
    print("main:", git("log", "--oneline", "-1", cwd=CLEAN).decode().strip())
    if PUSH:
        r = subprocess.run(["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential", "-c", "http.postBuffer=524288000", "push", URL, "main:main"], cwd=str(CLEAN), capture_output=True)
        print((r.stdout + r.stderr).decode("utf-8", "replace").strip()[-400:])
