import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

class TopologyEvaluator:
    """
    Evaluates the Topology-Aware Dimensional Error (E_topo) for fine-framed structures.
    """
    def __init__(self, physical_gt_diameter: float, lambda_weight: float = 1.0):
        """
        Args:
            physical_gt_diameter: 真实的物理卡尺测量值 (比如管子的直径，单位与点云一致)
            lambda_weight: 衡量管径收缩误差与节点位移误差的平衡系数
        """
        self.gt_diameter = physical_gt_diameter
        self.lambda_weight = lambda_weight

    def estimate_local_diameter(self, pcd: o3d.geometry.PointCloud, search_radius: float):
        """
        利用局部 PCA 计算沿骨架方向的法平面截面半径，从而推导直径。
        这能极其敏锐地发现 PointCleanNet 带来的"结构收缩(Over-averaging)"现象。
        """
        points = np.asarray(pcd.points)
        tree = cKDTree(points)
        diameters = []

        # 降采样提取骨架节点作为采样点
        skeleton_nodes = pcd.voxel_down_sample(voxel_size=search_radius).points
        
        for node in skeleton_nodes:
            # 找局部邻域
            idx = tree.query_ball_point(node, r=search_radius)
            if len(idx) < 10:
                continue
                
            local_patch = points[idx]
            
            # PCA 找管线的延伸方向 (主成分) 和截面方向
            pca = PCA(n_components=3)
            pca.fit(local_patch)
            
            # pca.components_[0] 是管子的走向
            # pca.components_[1] 和 pca.components_[2] 构成了横截面
            # 将点投影到横截面上计算分布的边界，这就是物理直径
            projected_2d = pca.transform(local_patch)[:, 1:] 
            
            # 使用 95% 分位数计算半径，避免极端飞线噪声干扰
            radius = np.percentile(np.linalg.norm(projected_2d, axis=1), 95)
            diameters.append(radius * 2.0)
            
        if not diameters:
            return float('inf') # 拓扑彻底断裂，返回无限大惩罚
            
        return np.mean(diameters)

    def evaluate(self, pred_pcd: o3d.geometry.PointCloud, gt_junctions: np.ndarray):
        """
        计算 E_topo
        Args:
            pred_pcd: 网络输出的点云
            gt_junctions: 真实的接头/拐角坐标 (K, 3)
        Returns:
            dict: 包含各项指标的字典
        """
        points = np.asarray(pred_pcd.points)
        tree = cKDTree(points)
        
        # 1. 计算 E_junc (拐角节点漂移误差)
        # 看看网络的过度平滑是不是把 T 型接头的直角拉成了圆弧
        junc_errors = []
        for gt_junc in gt_junctions:
            # 找到点云中距离真实接头最近的实体点
            dist, _ = tree.query(gt_junc)
            junc_errors.append(dist)
        e_junc = np.mean(junc_errors)
        
        # 2. 计算 E_caliper (卡尺管径收缩误差)
        # 取搜索半径为真实直径的 1.5 倍
        estimated_diam = self.estimate_local_diameter(pred_pcd, search_radius=self.gt_diameter * 1.5)
        e_caliper = np.abs(estimated_diam - self.gt_diameter)
        
        # 3. 综合 E_topo
        e_topo = e_junc + self.lambda_weight * e_caliper
        
        return {
            "E_topo": e_topo,
            "E_junc": e_junc,
            "E_caliper": e_caliper,
            "Estimated_Diameter": estimated_diam,
            "GT_Diameter": self.gt_diameter
        }

# --- 测试用例 ---
if __name__ == "__main__":
    # 模拟物理真实数据: 假设我们重建的管线标准直径是 50mm (0.05m)
    evaluator = TopologyEvaluator(physical_gt_diameter=0.05, lambda_weight=2.0)
    
    # 模拟网络吐出的假点云 (随机点云，管径会被测得很粗或极细)
    dummy_pcd = o3d.geometry.PointCloud()
    dummy_pcd.points = o3d.utility.Vector3dVector(np.random.rand(10000, 3) * 0.5)
    
    # 模拟两个关键 T型接头 的 GT 坐标
    gt_joints = np.array([[0.1, 0.1, 0.1], [0.4, 0.2, 0.3]])
    
    metrics = evaluator.evaluate(dummy_pcd, gt_joints)
    print("拓扑感知误差测试结果:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")