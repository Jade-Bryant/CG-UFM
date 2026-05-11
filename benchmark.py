#!/usr/bin/env python3
"""
CG-UFM Global Evaluation Benchmark
====================================
Evaluates multiple point cloud reconstruction methods against GT point clouds,
computes four metrics, and generates:
  - Console summary table  (mean ± std)
  - evaluation_results.csv
  - evaluation_table.tex   (booktabs, best values \\textbf{bolded})

Directory convention
--------------------
  datasets/test_gt/          ← GT .ply files, e.g. 0001.ply
                               Optional junction annotations: 0001_junctions.npy
  results/
    Ours/                    ← prediction .ply files with identical stem names
    PCN/
    Diffusion/
    Ours_no_UOT/

Usage
-----
  python benchmark.py \\
      --gt-dir      datasets/test_gt \\
      --results-dir results \\
      --output-dir  outputs/benchmark \\
      --gt-diameter 0.05 \\
      --workers     4

  # Evaluate a subset of methods only:
  python benchmark.py --methods Ours PCN
"""

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree
from tqdm import tqdm

# Ensure project root is importable regardless of CWD
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

METRICS: List[str] = ["CD", "CD_norm", "F-Score@1%", "F-Score@2%", "F-Score@5%",
                      "E_junc", "E_caliper"]

# True  → smaller value is better  (used to pick the "best" cell to bold)
LOWER_BETTER: Dict[str, bool] = {
    "CD":          True,
    "CD_norm":     True,
    "F-Score@1%":  False,
    "F-Score@2%":  False,
    "F-Score@5%":  False,
    "E_junc":      True,
    "E_caliper":   True,
}

# LaTeX column headers aligned with METRICS order
_LATEX_HEADERS = [
    r"CD $(\downarrow)$",
    r"CD$_{\text{norm}}$ $(\downarrow)$",
    r"F@1\% $(\uparrow)$",
    r"F@2\% $(\uparrow)$",
    r"F@5\% $(\uparrow)$",
    r"$E_{\text{junc}}$ $(\downarrow)$",
    r"$E_{\text{caliper}}$ $(\downarrow)$",
]


# ─────────────────────────────────────────────────────────────────────────────
# Pure metric functions  (module-level so they are picklable by ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric L2 Chamfer Distance.
    CD = mean_{p∈pred} min_{g∈gt} ||p-g||² + mean_{g∈gt} min_{p∈pred} ||g-p||²
    """
    tree_gt   = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _  = tree_gt.query(pred, k=1)
    d_g2p, _  = tree_pred.query(gt,   k=1)
    return float(np.mean(d_p2g ** 2) + np.mean(d_g2p ** 2))


def _normalize_to_unit_sphere(pred: np.ndarray, gt: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Map both clouds into a GT-defined unit-radius bounding sphere.

    Returns (pred_norm, gt_norm, diag) where diag is the original GT bbox
    diagonal length (so callers can convert back to physical units).

    Why this exists: the raw CD column mixes per-scene scales (diag ranges
    from 1.75 to 2.46 across our test set), so CD_mean is hard to read —
    a small CD on a large scene might still indicate worse reconstruction
    than a larger CD on a tiny scene. Normalizing makes CD comparable.

    We define the normalization from GT alone (not pred + gt jointly) so
    that a degenerate prediction can't artificially shrink the metric by
    deflating the joint bbox.
    """
    aabb_min = gt.min(axis=0)
    aabb_max = gt.max(axis=0)
    center = (aabb_min + aabb_max) / 2.0
    diag = float(np.linalg.norm(aabb_max - aabb_min))
    scale = max(diag / 2.0, 1e-9)
    return (pred - center) / scale, (gt - center) / scale, diag


def _fscore(pred: np.ndarray, gt: np.ndarray,
            tau: Optional[float] = None) -> Tuple[float, float, float]:
    """
    F-Score, Precision, and Recall at threshold tau.
    Default tau = 1% of the GT bounding-box diagonal (scale-adaptive).
    Returns (F-Score, Precision, Recall).
    """
    if tau is None:
        tau = 0.01 * float(np.linalg.norm(gt.max(axis=0) - gt.min(axis=0)))
    tree_gt   = KDTree(gt)
    tree_pred = KDTree(pred)
    d_p2g, _  = tree_gt.query(pred, k=1)
    d_g2p, _  = tree_pred.query(gt,   k=1)
    precision = float(np.mean(d_p2g < tau))
    recall    = float(np.mean(d_g2p < tau))
    denom = precision + recall
    fscore = 2.0 * precision * recall / denom if denom > 1e-8 else 0.0
    return float(fscore), precision, recall


def _evaluate_pair_worker(args: tuple) -> dict:
    """
    Module-level entry point for ProcessPoolExecutor.
    All arguments must be serialisable primitives (no o3d objects).

    Args (unpacked from tuple):
        gt_str      : str path to GT .ply
        pred_str    : str path to prediction .ply
        junc_str    : str path to _junctions.npy  or  None
        gt_diameter : float  physical GT pipe diameter for E_caliper
        fscore_tau  : float  fixed F-Score threshold  or  None

    Returns:
        dict with keys = METRICS + ['sample']
        Failed metrics are set to float('nan').
    """
    # Re-insert path inside worker in case of multiprocessing spawn context
    _root = str(Path(__file__).resolve().parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from metrics.topo_eval import TopologyEvaluator  # local import for fork safety

    gt_str, pred_str, junc_str, gt_diameter, fscore_tau = args
    result: dict = {m: float("nan") for m in METRICS}
    result["sample"] = Path(pred_str).stem

    # Per-scene physical diameter override. cad_to_gt writes a per-mesh
    # diameter estimate into the .pt; stage_infer copies it next to the GT
    # .ply as <stem>_diameter.npy. Falling back to the global --gt-diameter
    # is fine for legacy GT dirs that don't have the sidecar.
    diameter_sidecar = Path(gt_str).with_name(Path(gt_str).stem + "_diameter.npy")
    if diameter_sidecar.exists():
        try:
            gt_diameter = float(np.load(str(diameter_sidecar)))
        except Exception:
            pass  # keep the CLI fallback if the file is corrupt

    try:
        gt_pcd   = o3d.io.read_point_cloud(gt_str)
        pred_pcd = o3d.io.read_point_cloud(pred_str)

        if len(gt_pcd.points) == 0 or len(pred_pcd.points) == 0:
            warnings.warn(f"Empty cloud: {Path(pred_str).name}")
            return result

        gt_pts   = np.asarray(gt_pcd.points,   dtype=np.float64)
        pred_pts = np.asarray(pred_pcd.points,  dtype=np.float64)

        # ── Chamfer Distance ───────────────────────────────────────────────
        # Raw CD (in the original CAD frame's units — depends on per-scene scale).
        result["CD"] = _chamfer_distance(pred_pts, gt_pts)

        # Scale-invariant CD: both clouds normalized to a unit-radius sphere
        # defined by GT, so CD_norm is comparable across categories.
        pred_n, gt_n, gt_diag = _normalize_to_unit_sphere(pred_pts, gt_pts)
        result["CD_norm"] = _chamfer_distance(pred_n, gt_n)

        # ── F-Score @ multiple thresholds ──────────────────────────────────
        # τ as % of GT bbox diag. 1% is the standard but very tight for
        # long thin objects (a 2m pipe → τ=2cm); reporting 2% and 5% gives
        # a more complete picture and matches how Pointr / PCN report.
        # If --fscore-tau is provided we honor it for the canonical 1% slot
        # (legacy behaviour) and skip the multi-tau breakdown.
        if fscore_tau is not None:
            result["F-Score@1%"], _, _ = _fscore(pred_pts, gt_pts, tau=fscore_tau)
            # leave 2%/5% as NaN — caller chose to override
        else:
            for pct, key in [(0.01, "F-Score@1%"),
                             (0.02, "F-Score@2%"),
                             (0.05, "F-Score@5%")]:
                tau_i = pct * gt_diag
                result[key], _, _ = _fscore(pred_pts, gt_pts, tau=tau_i)

        # ── Topology metrics ───────────────────────────────────────────────
        # New TopoEvaluator: voxel-skeleton-graph based. E_caliper compares
        # pred-vs-GT radii at every skeleton node (catches over-smoothing);
        # E_junc bipartite-matches junctions (catches branch hallucination /
        # collapse). Skeleton + junctions are cached on disk as
        # <stem>_skeleton.npz / <stem>_junctions.npy next to the GT .ply,
        # so subsequent benchmark runs reuse them without re-extracting.
        gt_path_p = Path(gt_str)
        evaluator = TopologyEvaluator(
            physical_gt_diameter=gt_diameter,
            cache_dir=gt_path_p.parent,
            stem=gt_path_p.stem,
        )
        # Build (or load from sidecar) the GT skeleton + junctions.
        evaluator.build_or_load_gt_graph(gt_pcd, str(gt_path_p))

        # If the legacy junction sidecar was explicitly provided, honour it
        # (caller may have produced hand-curated junctions); otherwise the
        # evaluator falls back to the freshly extracted/loaded ones.
        gt_junctions_arg = None
        if junc_str is not None:
            try:
                gt_junctions_arg = np.load(junc_str)
            except Exception:
                gt_junctions_arg = None  # silently fall through to auto-extracted

        topo = evaluator.evaluate(pred_pcd, gt_junctions_arg, gt_pcd=gt_pcd)
        e_junc    = topo["E_junc"]
        e_caliper = topo["E_caliper"]
        result["E_junc"]    = float(e_junc)    if np.isfinite(e_junc)    else float("nan")
        result["E_caliper"] = float(e_caliper) if np.isfinite(e_caliper) else float("nan")

    except Exception as exc:  # never let one bad file crash the whole run
        warnings.warn(f"[WARN] {Path(pred_str).name}: {type(exc).__name__}: {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BenchmarkPipeline
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkPipeline:
    """
    Orchestrates end-to-end evaluation of multiple reconstruction methods.

    Parameters
    ----------
    gt_dir      : Path to GT .ply files (and optional _junctions.npy annotations).
    results_dir : Parent directory; each immediate sub-directory is one method.
    output_dir  : Where to write evaluation_results.csv and evaluation_table.tex.
    gt_diameter : Physical pipe diameter (metres) used for E_caliper baseline.
    n_workers   : Parallel worker count. 1 = sequential (useful for debugging).
    fscore_tau  : Fixed F-Score distance threshold; None → 1% GT bbox diagonal.
    """

    def __init__(
        self,
        gt_dir:      str,
        results_dir: str,
        output_dir:  str            = "outputs/benchmark",
        gt_diameter: float          = 0.05,
        n_workers:   int            = 4,
        fscore_tau:  Optional[float]= None,
    ) -> None:
        self.gt_dir      = Path(gt_dir)
        self.results_dir = Path(results_dir)
        self.output_dir  = Path(output_dir)
        self.gt_diameter = gt_diameter
        self.n_workers   = n_workers
        self.fscore_tau  = fscore_tau
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover_methods(self) -> List[str]:
        """Return sorted list of method names (immediate sub-directories of results_dir)."""
        return sorted(d.name for d in self.results_dir.iterdir() if d.is_dir())

    def _build_pairs(
        self, method_name: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Match prediction .ply files to GT .ply by stem name.
        Returns list of (gt_path_str, pred_path_str, junc_path_str_or_None).
        """
        method_dir = self.results_dir / method_name
        if not method_dir.exists():
            warnings.warn(f"Method directory not found: {method_dir}")
            return []

        pairs: List[Tuple[str, str, Optional[str]]] = []
        for pred_path in sorted(method_dir.glob("*.ply")):
            stem    = pred_path.stem
            gt_path = self.gt_dir / f"{stem}.ply"
            if not gt_path.exists():
                warnings.warn(f"[{method_name}] GT for '{stem}' not found — skipped.")
                continue
            junc_path = self.gt_dir / f"{stem}_junctions.npy"
            pairs.append((
                str(gt_path),
                str(pred_path),
                str(junc_path) if junc_path.exists() else None,
            ))

        if not pairs:
            warnings.warn(f"[{method_name}] No matching GT/prediction pairs found.")
        return pairs

    # ── Per-method evaluation ─────────────────────────────────────────────────

    def evaluate_single_method(self, method_name: str) -> pd.DataFrame:
        """
        Evaluate all samples for one method, optionally in parallel.

        Returns
        -------
        pd.DataFrame  indexed by sample stem,  columns = METRICS
        """
        pairs = self._build_pairs(method_name)
        if not pairs:
            return pd.DataFrame(columns=METRICS)

        worker_args = [
            (gt, pred, junc, self.gt_diameter, self.fscore_tau)
            for gt, pred, junc in pairs
        ]

        records: List[dict] = []

        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(_evaluate_pair_worker, arg): Path(arg[1]).stem
                    for arg in worker_args
                }
                pbar = tqdm(as_completed(futures), total=len(futures),
                            desc=f"  {method_name:<22}", ncols=80, leave=False)
                for future in pbar:
                    records.append(future.result())
        else:
            for arg in tqdm(worker_args, desc=f"  {method_name:<22}",
                            ncols=80, leave=False):
                records.append(_evaluate_pair_worker(arg))

        df = (
            pd.DataFrame(records)
            .set_index("sample")
            .reindex(columns=METRICS)   # guarantee column order even if all NaN
        )
        return df

    # ── All-method evaluation ─────────────────────────────────────────────────

    def evaluate_all_methods(
        self, methods: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate all discovered (or specified) methods.

        Returns
        -------
        dict  {method_name: per-sample DataFrame}
        """
        if methods is None:
            methods = self.discover_methods()
        if not methods:
            raise RuntimeError(f"No method directories found under {self.results_dir}")

        _sep = "═" * 62
        print(f"\n{_sep}")
        print(f"  CG-UFM Benchmark   |   {len(methods)} method(s)")
        print(f"  GT dir     : {self.gt_dir}")
        print(f"  Results dir: {self.results_dir}")
        print(f"  Workers    : {self.n_workers}")
        print(f"  GT diameter: {self.gt_diameter} m")
        print(f"{_sep}")

        all_results: Dict[str, pd.DataFrame] = {}
        for method in methods:
            print(f"\n▶  {method}")
            df = self.evaluate_single_method(method)
            all_results[method] = df
            n_valid = int(df.notna().any(axis=1).sum())
            print(f"   {n_valid} / {len(df)} samples evaluated successfully.")

        return all_results

    # ── Report ────────────────────────────────────────────────────────────────

    def _build_summary(
        self, all_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Aggregate per-sample results into mean / std / n per method × metric.
        Returns DataFrame indexed by method name.
        """
        rows = []
        for method, df in all_results.items():
            row: Dict[str, object] = {"Method": method}
            for metric in METRICS:
                valid = df[metric].dropna()
                row[f"{metric}_mean"] = valid.mean() if len(valid) else float("nan")
                row[f"{metric}_std"]  = valid.std()  if len(valid) else float("nan")
                row[f"{metric}_n"]    = int(len(valid))
            rows.append(row)
        return pd.DataFrame(rows).set_index("Method")

    def _best_per_metric(
        self, summary: pd.DataFrame
    ) -> Dict[str, Optional[str]]:
        """Return {metric: method_name_with_best_mean} (None if all NaN)."""
        best: Dict[str, Optional[str]] = {}
        for metric in METRICS:
            col   = summary[f"{metric}_mean"]
            valid = col.dropna()
            if valid.empty:
                best[metric] = None
            elif LOWER_BETTER[metric]:
                best[metric] = str(valid.idxmin())
            else:
                best[metric] = str(valid.idxmax())
        return best

    def generate_report(self, all_results: Dict[str, pd.DataFrame]) -> None:
        """Print console table, write CSV, write LaTeX table."""
        summary = self._build_summary(all_results)

        # ── Console table ─────────────────────────────────────────────────
        print(f"\n{'═'*62}")
        print("  Evaluation Summary  (mean ± std)")
        print(f"{'═'*62}")
        disp: Dict[str, List[str]] = {m: [] for m in METRICS}
        for method in summary.index:
            for metric in METRICS:
                mu  = summary.loc[method, f"{metric}_mean"]
                std = summary.loc[method, f"{metric}_std"]
                disp[metric].append(
                    "—" if pd.isna(mu) else f"{mu:.4f} ± {std:.4f}"
                )
        disp_df = pd.DataFrame(disp, index=summary.index)
        print(disp_df.to_string())

        # E_junc note if entirely missing
        if summary[[f"E_junc_n"]].max().item() == 0:
            print("\n  ⚠  E_junc: no _junctions.npy annotations found in GT dir.")
            print("     Place files as <stem>_junctions.npy to enable this metric.")

        # ── CSV ───────────────────────────────────────────────────────────
        csv_path = self.output_dir / "evaluation_results.csv"
        summary.to_csv(str(csv_path), float_format="%.6f")
        print(f"\n✔  CSV   → {csv_path}")

        # ── LaTeX ─────────────────────────────────────────────────────────
        latex    = self._generate_latex(summary)
        tex_path = self.output_dir / "evaluation_table.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"✔  LaTeX → {tex_path}")
        print(f"\n{'─'*62}")
        print("  LaTeX Table (copy-paste into your paper):")
        print(f"{'─'*62}\n")
        print(latex)

    def _generate_latex(self, summary: pd.DataFrame) -> str:
        """
        Build a booktabs LaTeX table.

        Rules:
          - One row per method, one column per metric.
          - Best value per metric is wrapped in \\textbf{}.
          - Metrics that are entirely NaN show '—' and are never bolded.
          - Underscores in method names are escaped.
          - Requires \\usepackage{booktabs} in the paper preamble.
        """
        best = self._best_per_metric(summary)

        col_spec      = "l" + "c" * len(METRICS)
        header_str    = " & ".join(_LATEX_HEADERS)

        lines = [
            "% ─────────────────────────────────────────────────────────",
            "% Generated by CG-UFM benchmark.py — paste into your paper.",
            "% Requires: \\usepackage{booktabs}",
            "% ─────────────────────────────────────────────────────────",
            r"\begin{table}[t]",
            r"  \centering",
            (
                r"  \caption{Quantitative comparison on underwater fine-framed structure "
                r"reconstruction. "
                r"Best results are \textbf{bolded}. "
                r"F-Score threshold $\tau = 1\%$ of GT bounding-box diagonal.}"
            ),
            r"  \label{tab:quantitative}",
            r"  \setlength{\tabcolsep}{6pt}",
            r"  \begin{tabular}{" + col_spec + r"}",
            r"    \toprule",
            f"    Method & {header_str} \\\\",
            r"    \midrule",
        ]

        for method in summary.index:
            safe_name = method.replace("_", r"\_")
            cells     = [safe_name]
            for metric in METRICS:
                mu = summary.loc[method, f"{metric}_mean"]
                if pd.isna(mu):
                    cells.append(r"—")
                else:
                    cell = f"{mu:.4f}"
                    if best.get(metric) == method:
                        cell = r"\textbf{" + cell + r"}"
                    cells.append(cell)
            lines.append("    " + " & ".join(cells) + r" \\")

        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",   # trailing newline
        ]

        return "\n".join(lines)

    # ── One-call entry point ──────────────────────────────────────────────────

    def run(self, methods: Optional[List[str]] = None) -> None:
        """Full pipeline: discover → evaluate → report."""
        all_results = self.evaluate_all_methods(methods)
        self.generate_report(all_results)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CG-UFM Global Benchmark Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python benchmark.py \\\n"
            "      --gt-dir datasets/test_gt \\\n"
            "      --results-dir results \\\n"
            "      --output-dir outputs/benchmark \\\n"
            "      --workers 4\n"
        ),
    )
    p.add_argument(
        "--gt-dir", default="datasets/test_gt",
        help="Directory containing GT .ply files (and optional _junctions.npy).",
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Parent directory; each sub-directory is one method.",
    )
    p.add_argument(
        "--output-dir", default="outputs/benchmark",
        help="Directory to write evaluation_results.csv and evaluation_table.tex.",
    )
    p.add_argument(
        "--gt-diameter", type=float, default=0.05,
        help="Physical GT pipe diameter in metres (used for E_caliper).",
    )
    p.add_argument(
        "--fscore-tau", type=float, default=None,
        help="Fixed F-Score distance threshold. Default: 1%% of GT bbox diagonal.",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes (1 = sequential, safer for debugging).",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Evaluate only these method names (default: all sub-directories).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = BenchmarkPipeline(
        gt_dir      = args.gt_dir,
        results_dir = args.results_dir,
        output_dir  = args.output_dir,
        gt_diameter = args.gt_diameter,
        n_workers   = args.workers,
        fscore_tau  = args.fscore_tau,
    )
    pipeline.run(methods=args.methods)


if __name__ == "__main__":
    main()
