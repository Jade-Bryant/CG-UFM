"""
Physical-aware metrics for the CG-UFM project.

Two metrics live here:

* **E_caliper** — Slides a virtual caliper along the GT skeleton and measures
  the radius of the perpendicular slab on *both* the predicted and the GT
  point clouds at every skeleton node, then reports the mean absolute
  difference. This is what catches "over-smoothing / pipe shrinkage": a
  predicted pipe that is uniformly thinner than the GT shows up directly as a
  positive E_caliper, regardless of overall bbox alignment.

* **E_junc** — Extracts junctions (degree ≥ 3 skeleton nodes) from both
  clouds, bipartite-matches them with the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`), then sums the matched-pair
  Chamfer distance with a degree-mismatch penalty. This catches both
  "branch hallucination" (extra junctions in pred) and "branch collapse"
  (T-joint flattened into a curve).

Both metrics rely on a voxel-skeleton graph (`_voxel_skeleton_graph`):
voxel-downsample the cloud, connect each voxel-node to its KNN neighbours
within a distance threshold, keep the largest connected component. This is
intentionally simple — comparing a *pred* skeleton against a *GT* skeleton
under the *same* extractor is what makes the metric well-defined, not the
absolute quality of either skeleton.

The class is exported under the legacy name ``TopologyEvaluator`` so
``benchmark.py`` doesn't need to learn a new symbol. The legacy
``estimate_local_diameter`` and ``evaluate(pred_pcd, gt_junctions)``
signatures are also preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
import networkx as nx
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _voxel_skeleton_graph(
    points: np.ndarray,
    voxel_size: float,
    k_neighbors: int = 6,
    edge_factor: float = 1.8,
) -> Tuple[np.ndarray, nx.Graph]:
    """Build a coarse skeleton graph from a dense point cloud.

    Pipeline:
      1. Voxel-downsample (Open3D) — each occupied voxel becomes a node.
      2. KNN-connect: each node links to its ``k_neighbors`` nearest peers,
         but only if the edge length is below ``edge_factor * voxel_size``.
         This keeps the graph following the surface and drops cross-pipe
         shortcut edges.
      3. Keep only the largest connected component — voxel noise can spawn
         tiny isolated islands that pollute the degree distribution.

    Args:
        points:       (N, 3) point cloud in metric units.
        voxel_size:   side length of the voxel filter; controls skeleton
                      resolution. Roughly ``gt_diameter * 1.5`` works well
                      for pipes (caliper-sized voxels).
        k_neighbors:  k for the KNN connectivity pass.
        edge_factor:  reject edges longer than ``edge_factor * voxel_size``.

    Returns:
        nodes:  (K, 3) voxel-centroid coordinates (after LCC filtering).
        graph:  networkx.Graph with K nodes indexed 0..K-1 and the kept
                edges. ``graph.degree[i]`` is the local branch count.

    Raises:
        ValueError: if the input is empty or voxel_size is non-positive.
    """
    if points.size == 0:
        raise ValueError("empty point cloud")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    downsampled = pcd.voxel_down_sample(voxel_size)
    nodes = np.asarray(downsampled.points, dtype=np.float64)

    if nodes.shape[0] < 2:
        # Degenerate: a single voxel can't form a graph. Return as-is with an
        # empty edge set — caller decides whether that's a fatal condition.
        return nodes, nx.Graph()

    tree = cKDTree(nodes)
    # query k+1 because the closest neighbour is always the node itself.
    k = min(k_neighbors + 1, nodes.shape[0])
    dists, idxs = tree.query(nodes, k=k)
    edge_threshold = edge_factor * voxel_size

    g = nx.Graph()
    g.add_nodes_from(range(nodes.shape[0]))
    for i in range(nodes.shape[0]):
        for j_pos in range(1, k):  # skip self
            j = int(idxs[i, j_pos])
            d = float(dists[i, j_pos])
            if d <= edge_threshold:
                g.add_edge(i, j, weight=d)

    # Keep only the largest connected component
    if g.number_of_edges() > 0:
        components = list(nx.connected_components(g))
        if components:
            largest = max(components, key=len)
            keep = sorted(largest)
            sub = g.subgraph(keep).copy()
            # Re-index 0..K'-1 so the returned node-array stays aligned
            relabel = {old: new for new, old in enumerate(keep)}
            sub = nx.relabel_nodes(sub, relabel)
            nodes = nodes[keep]
            g = sub

    return nodes, g


def _local_tangent(nodes: np.ndarray, graph: nx.Graph, idx: int) -> Optional[np.ndarray]:
    """Estimate the local pipe tangent at ``nodes[idx]``.

    Uses the average direction to graph neighbours. For a degree-1 leaf this
    is just the single edge direction; for a degree-2 mid-pipe node it
    averages the two opposite directions (signs flipped so they don't
    cancel); for a junction we fall back to local PCA on neighbours since the
    "tangent" concept is ambiguous there.

    Returns:
        (3,) unit vector, or None if the tangent is degenerate (no neighbours
        or zero-length direction).
    """
    neigh = list(graph.neighbors(idx))
    if not neigh:
        return None

    center = nodes[idx]
    diffs = nodes[neigh] - center  # (deg, 3)

    if len(neigh) >= 3:
        # Junction: use PCA over signed neighbour offsets — first component
        # is the dominant axis through this junction.
        try:
            pca = PCA(n_components=1)
            pca.fit(diffs)
            t = pca.components_[0]
        except Exception:
            t = diffs.mean(axis=0)
    elif len(neigh) == 2:
        # Mid-pipe: flip one so they reinforce, then average.
        a, b = diffs[0], diffs[1]
        if np.dot(a, b) < 0:
            t = a - b
        else:
            t = a + b
    else:
        t = diffs[0]

    norm = np.linalg.norm(t)
    if norm < 1e-9:
        return None
    return t / norm


def _slab_radius(
    tree: cKDTree,
    points: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    slab_thickness: float,
    max_radius: float,
    min_pts: int = 5,
) -> float:
    """Mean Euclidean distance from ``center`` to slab-inlier points.

    A "slab" is the band of points whose projection onto ``tangent`` lies
    within ``±slab_thickness/2`` of the center, AND whose 3D distance to
    center is below ``max_radius`` (to avoid grabbing a parallel branch in
    a T-joint).

    Returns:
        mean radius (positive float), or NaN if fewer than ``min_pts`` points
        survive the filter.
    """
    cand_idx = tree.query_ball_point(center, r=max_radius)
    if len(cand_idx) < min_pts:
        return float("nan")
    cand = points[cand_idx]
    offsets = cand - center                              # (n, 3)
    along = offsets @ tangent                            # signed projection
    in_slab = np.abs(along) < (slab_thickness * 0.5)
    if in_slab.sum() < min_pts:
        return float("nan")
    slab_pts = offsets[in_slab]
    radii = np.linalg.norm(slab_pts, axis=1)
    return float(radii.mean())


def _chamfer_sym(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean-nearest-neighbour Chamfer distance.

    Returns NaN if either set is empty (caller decides what to do with that).
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_ab, _ = tb.query(a)
    d_ba, _ = ta.query(b)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class TopoEvaluator:
    """Physics-aware geometric evaluator for fine-framed pipe reconstructions.

    Constructed once per scene; caches the GT skeleton on disk so subsequent
    runs (and the multi-method benchmark passes) don't re-extract it.

    Args:
        physical_gt_diameter: Nominal pipe diameter for this scene (used both
            as a voxel-size heuristic and as a backward-compatible reference
            for the legacy ``estimate_local_diameter`` method).
        voxel_size: Skeleton resolution. Defaults to
            ``physical_gt_diameter * 1.5``.
        slab_thickness: Caliper slab thickness when computing E_caliper.
            Defaults to ``voxel_size * 0.5``.
        max_radius_factor: Maximum slab radius in units of pipe diameter.
            Defaults to 3.0 — large enough to capture a fat T-joint, small
            enough to ignore a parallel branch one pipe-width away.
        junc_degree_threshold: Minimum node degree to count as a junction.
            Defaults to 3 (a T-joint has degree 3, an X-joint has 4).
        lambda_degree: Weight on the degree-mismatch term in E_junc.
        penalty_max: Numerical value returned when a metric is undefined
            (e.g. predicted cloud empty). Kept finite so np.isfinite passes
            and benchmark.py can still aggregate.
        cache_dir: Directory for sidecar files. If None, sidecar caching is
            disabled.
        stem: Filename stem used for sidecar lookup. Required when
            ``cache_dir`` is set.

    The legacy alias ``TopologyEvaluator`` is also exported for backward
    compatibility with ``benchmark.py:161``.
    """

    def __init__(
        self,
        physical_gt_diameter: float,
        voxel_size: Optional[float] = None,
        slab_thickness: Optional[float] = None,
        max_radius_factor: float = 3.0,
        junc_degree_threshold: int = 3,
        lambda_degree: float = 1.0,
        lambda_weight: Optional[float] = None,  # legacy alias for lambda_degree
        penalty_max: float = 1.0,
        cache_dir: Optional[Path] = None,
        stem: Optional[str] = None,
    ):
        if physical_gt_diameter <= 0:
            raise ValueError(
                f"physical_gt_diameter must be > 0, got {physical_gt_diameter}"
            )
        self.gt_diameter = float(physical_gt_diameter)
        self.voxel_size = float(voxel_size) if voxel_size else self.gt_diameter * 1.5
        self.slab_thickness = (
            float(slab_thickness) if slab_thickness else self.voxel_size * 0.5
        )
        self.max_radius = self.gt_diameter * max_radius_factor
        self.junc_degree_threshold = int(junc_degree_threshold)
        # Accept the legacy ``lambda_weight`` kwarg if a caller still uses it.
        self.lambda_degree = float(
            lambda_weight if lambda_weight is not None else lambda_degree
        )
        self.penalty_max = float(penalty_max)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.stem = stem

        # Populated by build_or_load_gt_graph()
        self._gt_nodes: Optional[np.ndarray] = None
        self._gt_graph: Optional[nx.Graph] = None
        self._gt_junctions: Optional[np.ndarray] = None
        self._gt_junction_degrees: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────
    # Legacy API — kept for backward compatibility with benchmark.py
    # ─────────────────────────────────────────────────────────────────
    def estimate_local_diameter(
        self, pcd: o3d.geometry.PointCloud, search_radius: float
    ) -> float:
        """Estimate the average local diameter via PCA on voxel patches.

        This is the *legacy* estimator — kept because ``benchmark.py`` calls
        it as a fallback when no junctions are available. The new
        ``evaluate`` path no longer uses it. The 95th-percentile cross-section
        radius behaves badly on Reducers (variable diameter) so we use the
        mean instead, but the signature matches the old code.
        """
        points = np.asarray(pcd.points)
        if points.shape[0] == 0:
            return float("inf")
        tree = cKDTree(points)
        diameters = []
        skeleton_nodes = np.asarray(
            pcd.voxel_down_sample(search_radius).points
        )
        for node in skeleton_nodes:
            idx = tree.query_ball_point(node, r=search_radius)
            if len(idx) < 10:
                continue
            local_patch = points[idx]
            try:
                pca = PCA(n_components=3)
                pca.fit(local_patch)
            except Exception:
                continue
            projected_2d = pca.transform(local_patch)[:, 1:]
            radius = np.percentile(np.linalg.norm(projected_2d, axis=1), 95)
            diameters.append(radius * 2.0)
        if not diameters:
            return float("inf")
        return float(np.mean(diameters))

    # ─────────────────────────────────────────────────────────────────
    # New API
    # ─────────────────────────────────────────────────────────────────
    def build_or_load_gt_graph(
        self,
        gt_pcd: o3d.geometry.PointCloud,
        gt_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, nx.Graph, np.ndarray]:
        """Extract or load the GT skeleton + junctions.

        First call writes ``<stem>_skeleton.npz`` and ``<stem>_junctions.npy``
        sidecars next to the GT .ply (if ``cache_dir`` is set). Subsequent
        calls within the same process — or across processes once the sidecar
        is written — reuse the cached arrays.

        Args:
            gt_pcd:  Open3D point cloud of the GT (used only if we need to
                     rebuild).
            gt_path: Optional explicit GT .ply path, takes precedence over
                     the ``cache_dir``/``stem`` pair if provided.

        Returns:
            (gt_nodes, gt_graph, gt_junctions). ``gt_junctions`` may be empty
            for genuinely junction-free scenes (e.g. StraightPipe).
        """
        skel_path, junc_path = self._sidecar_paths(gt_path)

        # Try cache
        if skel_path is not None and skel_path.exists() and junc_path is not None and junc_path.exists():
            try:
                cached = np.load(str(skel_path), allow_pickle=False)
                nodes = cached["nodes"]
                edges = cached["edges"]
                g = nx.Graph()
                g.add_nodes_from(range(nodes.shape[0]))
                g.add_edges_from([tuple(e) for e in edges])
                junctions = np.load(str(junc_path))
                self._gt_nodes, self._gt_graph, self._gt_junctions = (
                    nodes, g, junctions
                )
                self._gt_junction_degrees = np.array(
                    [self._degree_at_point(g, nodes, p) for p in junctions],
                    dtype=np.int32,
                )
                return nodes, g, junctions
            except Exception as exc:
                warnings.warn(
                    f"[topo_eval] sidecar load failed ({exc}); rebuilding"
                )

        # Build from scratch
        gt_points = np.asarray(gt_pcd.points)
        nodes, g = _voxel_skeleton_graph(gt_points, self.voxel_size)
        if nodes.shape[0] == 0:
            self._gt_nodes = nodes
            self._gt_graph = g
            self._gt_junctions = np.zeros((0, 3))
            self._gt_junction_degrees = np.zeros((0,), dtype=np.int32)
            return nodes, g, self._gt_junctions

        deg = np.array([g.degree[i] for i in range(nodes.shape[0])], dtype=np.int32)
        junc_mask = deg >= self.junc_degree_threshold
        junctions = nodes[junc_mask]
        junction_degrees = deg[junc_mask]

        self._gt_nodes = nodes
        self._gt_graph = g
        self._gt_junctions = junctions
        self._gt_junction_degrees = junction_degrees

        # Write cache
        if skel_path is not None and junc_path is not None:
            try:
                skel_path.parent.mkdir(parents=True, exist_ok=True)
                edges_arr = (
                    np.asarray(list(g.edges()), dtype=np.int64)
                    if g.number_of_edges() > 0
                    else np.zeros((0, 2), dtype=np.int64)
                )
                np.savez(
                    str(skel_path),
                    nodes=nodes.astype(np.float64),
                    edges=edges_arr,
                    degrees=deg,
                )
                np.save(str(junc_path), junctions.astype(np.float64))
            except Exception as exc:
                warnings.warn(f"[topo_eval] sidecar save failed: {exc}")

        return nodes, g, junctions

    def evaluate(
        self,
        pred_pcd: o3d.geometry.PointCloud,
        gt_junctions: Optional[np.ndarray] = None,
        gt_pcd: Optional[o3d.geometry.PointCloud] = None,
    ) -> dict:
        """Compute E_caliper and E_junc for a single (pred, gt) pair.

        Caller is expected to have invoked ``build_or_load_gt_graph`` first
        so the GT skeleton is cached on ``self``. If not, and ``gt_pcd`` is
        provided, the build runs implicitly here.

        Args:
            pred_pcd:     Open3D point cloud of the prediction.
            gt_junctions: (J, 3) array of GT junction coordinates. If None,
                          falls back to the cached ones from
                          ``build_or_load_gt_graph``. Kept positional for
                          backward compat with the old signature.
            gt_pcd:       Optional. Required only if neither
                          ``build_or_load_gt_graph`` has been called nor any
                          GT skeleton has been cached on this instance.

        Returns:
            dict with at least the following keys (always present, may be
            ``nan`` or ``penalty_max`` on edge cases):
              - ``E_caliper``        (primary): pred-vs-gt radius mismatch.
              - ``E_junc``           (primary): position + degree mismatch.
              - ``E_caliper_nominal``: pred-vs-CAD-scalar radius mismatch,
                                       reproduces the legacy interpretation.
              - ``E_topo``           : E_junc + lambda * E_caliper (legacy
                                       composite, kept so old callers don't
                                       break).
              - ``n_junc_gt``, ``n_junc_pred``: counts.
              - ``junc_chamfer``, ``degree_mismatch``: components of E_junc.
              - ``Estimated_Diameter``, ``GT_Diameter``: legacy keys.
        """
        result = {
            "E_caliper": float("nan"),
            "E_junc": float("nan"),
            "E_caliper_nominal": float("nan"),
            "E_topo": float("nan"),
            "n_junc_gt": 0,
            "n_junc_pred": 0,
            "junc_chamfer": float("nan"),
            "degree_mismatch": float("nan"),
            "Estimated_Diameter": float("nan"),
            "GT_Diameter": self.gt_diameter,
        }

        pred_points = np.asarray(pred_pcd.points)
        if pred_points.shape[0] == 0:
            warnings.warn("[topo_eval] empty prediction cloud")
            result["E_caliper"] = self.penalty_max
            result["E_junc"] = self.penalty_max
            result["E_topo"] = self.penalty_max * (1.0 + self.lambda_degree)
            return result

        # Ensure GT graph is available (lazy build if caller forgot).
        if self._gt_nodes is None:
            if gt_pcd is None:
                raise RuntimeError(
                    "GT skeleton not built. Call build_or_load_gt_graph() "
                    "first, or pass gt_pcd to evaluate()."
                )
            self.build_or_load_gt_graph(gt_pcd)

        gt_nodes = self._gt_nodes
        gt_graph = self._gt_graph

        # Honour an externally supplied junction set if non-empty (preserves
        # the original API where callers could load _junctions.npy themselves).
        if gt_junctions is not None and len(gt_junctions) > 0:
            cached_juncs = np.asarray(gt_junctions, dtype=np.float64)
            cached_degrees = np.array(
                [self._degree_at_point(gt_graph, gt_nodes, p) for p in cached_juncs],
                dtype=np.int32,
            )
        else:
            cached_juncs = self._gt_junctions
            cached_degrees = self._gt_junction_degrees

        # ── E_caliper ─────────────────────────────────────────────────────
        if gt_pcd is not None:
            gt_points = np.asarray(gt_pcd.points)
        else:
            # If caller didn't pass gt_pcd, reconstruct from the cached
            # graph's nodes — coarser but consistent. Prefer the real GT
            # when available; benchmark.py always has it.
            gt_points = gt_nodes
        e_cal, e_cal_nominal = self._e_caliper(pred_points, gt_points, gt_nodes, gt_graph)
        result["E_caliper"] = e_cal
        result["E_caliper_nominal"] = e_cal_nominal

        # ── E_junc ─────────────────────────────────────────────────────────
        pred_nodes, pred_graph = _voxel_skeleton_graph(pred_points, self.voxel_size)
        if pred_nodes.shape[0] == 0:
            warnings.warn("[topo_eval] pred skeleton empty")
            result["E_junc"] = self.penalty_max
            result["n_junc_gt"] = int(len(cached_juncs))
            result["n_junc_pred"] = 0
            result["E_topo"] = self._topo(result["E_caliper"], result["E_junc"])
            return result

        pred_deg = np.array(
            [pred_graph.degree[i] for i in range(pred_nodes.shape[0])],
            dtype=np.int32,
        )
        pred_junc_mask = pred_deg >= self.junc_degree_threshold
        pred_junctions = pred_nodes[pred_junc_mask]
        pred_junction_degrees = pred_deg[pred_junc_mask]

        result["n_junc_gt"] = int(len(cached_juncs))
        result["n_junc_pred"] = int(len(pred_junctions))
        e_junc, junc_chamfer, deg_mis = self._e_junc(
            pred_junctions, pred_junction_degrees,
            cached_juncs, cached_degrees,
        )
        result["E_junc"] = e_junc
        result["junc_chamfer"] = junc_chamfer
        result["degree_mismatch"] = deg_mis

        # Legacy composite for callers that read E_topo.
        result["E_topo"] = self._topo(result["E_caliper"], result["E_junc"])

        # Legacy diameter key (kept finite for backward compat dashboards).
        result["Estimated_Diameter"] = self.estimate_local_diameter(
            pred_pcd, search_radius=self.gt_diameter * 1.5
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Implementation details
    # ─────────────────────────────────────────────────────────────────
    def _sidecar_paths(self, gt_path: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
        if gt_path is not None:
            base = Path(gt_path)
            stem = base.stem
            parent = base.parent
        elif self.cache_dir is not None and self.stem is not None:
            stem = self.stem
            parent = self.cache_dir
        else:
            return None, None
        return parent / f"{stem}_skeleton.npz", parent / f"{stem}_junctions.npy"

    @staticmethod
    def _degree_at_point(graph: nx.Graph, nodes: np.ndarray, point: np.ndarray) -> int:
        """Find the graph node closest to ``point`` and return its degree.

        Used to look up degrees for externally supplied junction coordinates
        (e.g. legacy ``_junctions.npy`` from the old CLI workflow).
        """
        if nodes.shape[0] == 0:
            return 0
        d = np.linalg.norm(nodes - point[None, :], axis=1)
        idx = int(d.argmin())
        return int(graph.degree[idx])

    def _e_caliper(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        gt_nodes: np.ndarray,
        gt_graph: nx.Graph,
    ) -> Tuple[float, float]:
        """Slide a caliper along the GT skeleton; return (pred-vs-gt, pred-vs-nominal)."""
        if gt_nodes.shape[0] == 0:
            return float("nan"), float("nan")

        pred_tree = cKDTree(pred_points)
        gt_tree = cKDTree(gt_points)
        nominal_radius = self.gt_diameter * 0.5

        diffs = []          # |r_pred − r_gt| per valid node
        nominal_diffs = []  # |r_pred − nominal|
        for k in range(gt_nodes.shape[0]):
            tangent = _local_tangent(gt_nodes, gt_graph, k)
            if tangent is None:
                continue
            center = gt_nodes[k]
            r_pred = _slab_radius(pred_tree, pred_points, center, tangent,
                                  self.slab_thickness, self.max_radius)
            r_gt = _slab_radius(gt_tree, gt_points, center, tangent,
                                self.slab_thickness, self.max_radius)
            if np.isfinite(r_pred) and np.isfinite(r_gt):
                diffs.append(abs(r_pred - r_gt))
            if np.isfinite(r_pred):
                nominal_diffs.append(abs(r_pred - nominal_radius))

        e_cal = float(np.mean(diffs)) if diffs else float("nan")
        e_cal_nominal = float(np.mean(nominal_diffs)) if nominal_diffs else float("nan")
        return e_cal, e_cal_nominal

    def _e_junc(
        self,
        pred_junctions: np.ndarray,
        pred_degrees: np.ndarray,
        gt_junctions: np.ndarray,
        gt_degrees: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute (E_junc, chamfer_term, degree_mismatch_term).

        Handles all four count combinations of (|J_pred|, |J_gt|):

          * both empty  → 0, 0, 0   (correctly identifies an empty StraightPipe).
          * pred empty, gt non-empty → penalty_max (catastrophic miss).
          * pred non-empty, gt empty → degree-mismatch penalty for hallucinations.
          * both non-empty → bipartite-matched Chamfer + |deg| sum.
        """
        n_pred = len(pred_junctions)
        n_gt = len(gt_junctions)

        if n_pred == 0 and n_gt == 0:
            return 0.0, 0.0, 0.0

        if n_pred == 0 and n_gt > 0:
            # Catastrophic: model missed every junction.
            return self.penalty_max, self.penalty_max, float(gt_degrees.sum())

        if n_gt == 0 and n_pred > 0:
            # Hallucinated joints where there are none.
            # No Chamfer to compute; cost is pure degree penalty.
            deg_pen = float(pred_degrees.sum())
            normalized = deg_pen / max(1, n_pred)
            return self.lambda_degree * normalized, 0.0, deg_pen

        # Both non-empty: Hungarian assignment on Euclidean cost.
        cost = np.linalg.norm(
            pred_junctions[:, None, :] - gt_junctions[None, :, :], axis=-1
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_distances = cost[row_ind, col_ind]
        matched_deg_diff = np.abs(
            pred_degrees[row_ind] - gt_degrees[col_ind]
        ).astype(np.float64)

        chamfer_term = _chamfer_sym(pred_junctions, gt_junctions)
        # Unmatched penalty: extra junctions on either side that the
        # Hungarian had to leave out (size mismatch). Each unmatched
        # contributes its own degree as the imbalance.
        n_matched = len(row_ind)
        unmatched_pred = n_pred - n_matched
        unmatched_gt = n_gt - n_matched
        unmatched_penalty = float(unmatched_pred + unmatched_gt)

        deg_term = float(matched_deg_diff.sum() + unmatched_penalty)
        # Normalize the degree term by the GT junction count so it lives on
        # roughly the same scale as the Chamfer term (which is already in
        # metric units of the cloud).
        deg_term_normalized = deg_term / max(1, n_gt)

        e_junc = chamfer_term + self.lambda_degree * deg_term_normalized
        return e_junc, chamfer_term, deg_term

    def _topo(self, e_cal: float, e_junc: float) -> float:
        """Legacy composite E_topo = E_junc + lambda * E_caliper."""
        if not (np.isfinite(e_cal) and np.isfinite(e_junc)):
            return float("nan")
        return e_junc + self.lambda_degree * e_cal


# Legacy alias — benchmark.py imports this name.
TopologyEvaluator = TopoEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — sanity-check on a synthetic T-joint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def cylinder_segment(p0, p1, radius, n_pts):
        """Sample n_pts on the lateral surface of a cylinder between p0 and p1."""
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        axis = p1 - p0
        length = np.linalg.norm(axis)
        axis_unit = axis / length
        # Pick two orthogonal vectors in the cross-section plane.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, axis_unit)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis_unit, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(axis_unit, u)
        ts = rng.uniform(0, 1, size=n_pts)
        thetas = rng.uniform(0, 2 * np.pi, size=n_pts)
        return (
            p0
            + ts[:, None] * axis[None, :]
            + radius * (np.cos(thetas)[:, None] * u + np.sin(thetas)[:, None] * v)
        )

    def t_joint(radius, n_pts_per_branch=1500):
        a = cylinder_segment([-1, 0, 0], [1, 0, 0], radius, n_pts_per_branch)
        b = cylinder_segment([0, 0, 0], [0, 1, 0], radius, n_pts_per_branch)
        return np.vstack([a, b])

    print("─" * 60)
    print("Self-test 1: pred == gt → E_caliper, E_junc ≈ 0")
    print("─" * 60)
    radius = 0.1
    gt_pts = t_joint(radius)
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(gt_pts.copy())
    ev = TopoEvaluator(physical_gt_diameter=2 * radius)
    ev.build_or_load_gt_graph(gt_pcd)
    r = ev.evaluate(pred_pcd, gt_pcd=gt_pcd)
    for k, v in r.items():
        print(f"  {k:24s} {v}")

    print()
    print("─" * 60)
    print("Self-test 2: pred shrunk to 80% radius → E_caliper ≈ 0.2 * radius")
    print("─" * 60)
    pred_pts = t_joint(radius * 0.8)
    pred_pcd2 = o3d.geometry.PointCloud()
    pred_pcd2.points = o3d.utility.Vector3dVector(pred_pts)
    r2 = ev.evaluate(pred_pcd2, gt_pcd=gt_pcd)
    print(f"  E_caliper={r2['E_caliper']:.4f}  (expect ≈ {0.2*radius:.4f})")
    print(f"  E_junc   ={r2['E_junc']:.4f}")
    print(f"  n_junc_gt={r2['n_junc_gt']} n_junc_pred={r2['n_junc_pred']}")

    print()
    print("─" * 60)
    print("Self-test 3: pred missing the side branch → degree mismatch spikes")
    print("─" * 60)
    straight_pts = cylinder_segment([-1, 0, 0], [1, 0, 0], radius, 3000)
    pred_pcd3 = o3d.geometry.PointCloud()
    pred_pcd3.points = o3d.utility.Vector3dVector(straight_pts)
    r3 = ev.evaluate(pred_pcd3, gt_pcd=gt_pcd)
    print(f"  E_junc={r3['E_junc']:.4f}  (expect > self-test 1's value)")
    print(f"  junc_chamfer={r3['junc_chamfer']}  degree_mismatch={r3['degree_mismatch']}")
    print(f"  n_junc_gt={r3['n_junc_gt']} n_junc_pred={r3['n_junc_pred']}")
    print("OK")
