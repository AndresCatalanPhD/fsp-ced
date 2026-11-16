"""Shared helpers for the experiment runners: resumable JSONL output, code version, session budget, sharding.

Resuming: every finished run is appended at once (and flushed to disk) to a JSONL file; when a runner starts, it
skips every (instance, variant, method[, seed]) already stored with the current version of that method
(CODE_VERSION, or METHOD_VERSION for the methods that changed later).  Records written by an
older version of the code are kept in the file but ignored, so those runs are repeated.  A run interrupted by a
disconnection is simply repeated.

Session budget: if the environment variables FSP_T0 (start time, epoch seconds) and FSP_BUDGET_H (hours) are set, a
runner stops launching new runs once the budget is spent, so that everything is saved before Colab ends the session.
"""
import glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CODE_VERSION = "1"
# Methods whose code changed after the first release are versioned on their own, so that only they are rerun.
# Version 3: decomposition with the flow-shop subproblem, Johnson and ladder inequalities and deficit cuts (LBBD and
# its ablation variants), and the flow-shop CP model (CPF).  Records of the decomposition written by versions 1-2
# are ignored.
METHOD_VERSION = {"LBBD": "3", "LBBD-B": "3", "LBBD-P": "3", "LBBD-N": "3", "LBBD-S": "3", "LBBD-0": "4",
                  "LBBD-T": "3", "LBBD-BN": "3", "CPF": "3", "CPF0": "3", "CPS": "3", "THEORY": "3",
                  "ROOT": "4", "SENS": "4", "IG+CPF": "4"}


def version_of(method):
    return METHOD_VERSION.get(method, CODE_VERSION)


def instance_files(idir, shard="0/1", by_size=True, limit=0, pick=""):
    """Instance files of a folder, sorted by size (smallest first), restricted to a shard.  For quick tests:
    limit=N keeps the N smallest files of the shard, and pick='a,b' keeps the files whose names contain a or b."""
    files = sorted(f for f in glob.glob(os.path.join(idir, "*.json"))
                   if not os.path.basename(f).startswith("optima"))
    if by_size:
        files.sort(key=lambda f: (os.path.getsize(f), f))
    k, K = (int(x) for x in shard.split("/"))
    files = [f for q, f in enumerate(files) if q % K == k]
    if pick:
        keys = [x for x in pick.split(",") if x]
        files = [f for f in files if any(x in os.path.basename(f) for x in keys)]
    if limit and limit > 0:
        files = sorted(files, key=lambda f: (os.path.getsize(f), f))[:limit]
    return files


def done_keys(path, fields):
    s = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:      # a line cut by a disconnection
                    continue
                if str(r.get("code_version")) == version_of(r.get("method")):
                    s.add(tuple(r.get(f) for f in fields))
    return s


def append(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec = dict(rec, code_version=version_of(rec.get("method")), host=f"{os.uname().nodename}-{os.cpu_count()}cpu",
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        os.fsync(fo.fileno())


def budget_ok():
    t0, b = os.environ.get("FSP_T0"), os.environ.get("FSP_BUDGET_H")
    if not t0 or not b:
        return True
    if (time.time() - float(t0)) / 3600.0 < float(b):
        return True
    print("session budget spent: no new runs are launched (run the cell again in a new session to continue)",
          flush=True)
    return False
