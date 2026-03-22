import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

benchmarks = [
    "benchmark_learning_rate.py",
    "benchmark_batch_size.py",
    "benchmark_layer_size.py",
    "benchmark_num_conv_layers.py",
    "benchmark_num_dense_layers.py",
    "benchmark_activation.py",
    "benchmark_scheduler.py",
]

env = os.environ.copy()
env["MPLBACKEND"] = "Agg"       # save PNGs without opening windows
env["PYTHONPATH"] = str(PROJECT_ROOT)  # allow "from network.with_pytorch..." imports

benchmarks_dir = Path(__file__).parent

for filename in benchmarks:
    name = Path(filename).stem
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, str(benchmarks_dir / filename)],
        env=env,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"\n[ERREUR] {name} a échoué (code {result.returncode})")

print("\n\nTous les benchmarks sont terminés. PNGs sauvegardés dans network/with_pytorch/benchmarks/")
