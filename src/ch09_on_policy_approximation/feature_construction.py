"""
================================================================================
第9.5节：特征构造 - 创造良好的表示
Section 9.5: Feature Construction - Creating Good Representations
================================================================================

好的特征是成功的关键！
Good features are the key to success!

特征构造方法 Feature Construction Methods:
1. 多项式基 Polynomial Basis
2. 傅里叶基 Fourier Basis
3. 径向基函数 Radial Basis Functions
4. 瓦片编码 Tile Coding
5. 聚合器 Aggregators

关键思想 Key Ideas:
- 特征决定泛化
  Features determine generalization
- 局部vs全局特征
  Local vs global features
- 计算vs表达能力权衡
  Computation vs expressiveness tradeoff

瓦片编码优势 Tile Coding Advantages:
- 计算效率高
  Computationally efficient
- 稀疏表示
  Sparse representation
- 可控泛化
  Controllable generalization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import hashlib
import logging
from abc import ABC, abstractmethod

# 设置日志
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# 第9.5.1节：多项式特征
# Section 9.5.1: Polynomial Features
# ================================================================================

class PolynomialFeatures:
    """
    多项式基函数
    Polynomial Basis Functions
    
    将d维输入扩展为多项式特征
    Expand d-dimensional input to polynomial features
    
    例如 Example:
    输入 Input: [x₁, x₂]
    2次多项式 Degree 2: [1, x₁, x₂, x₁², x₁x₂, x₂²]
    
    特点 Characteristics:
    - 全局特征
      Global features
    - 适合光滑函数
      Good for smooth functions
    - 维度爆炸问题
      Curse of dimensionality
    """
    
    def __init__(self, degree: int = 2, include_bias: bool = True):
        """
        初始化多项式特征
        Initialize polynomial features
        
        Args:
            degree: 多项式度数
                   Polynomial degree
            include_bias: 是否包含偏置项(1)
                         Whether to include bias term
        """
        self.degree = degree
        self.include_bias = include_bias
        
        # 缓存特征索引
        # Cache feature indices
        self._feature_indices = None
        
        logger.info(f"初始化多项式特征: degree={degree}, bias={include_bias}")
    
    def fit(self, n_input_features: int):
        """
        计算特征映射
        Compute feature mapping
        
        Args:
            n_input_features: 输入特征数
                            Number of input features
        """
        # 生成所有可能的指数组合
        # Generate all possible exponent combinations
        indices = []
        
        def generate_indices(current, remaining_degree, start_idx):
            """递归生成指数组合"""
            if start_idx == n_input_features:
                if self.include_bias or sum(current) > 0:
                    indices.append(tuple(current))
                return
            
            for exp in range(remaining_degree + 1):
                current[start_idx] = exp
                generate_indices(current, remaining_degree - exp, start_idx + 1)
                current[start_idx] = 0
        
        generate_indices([0] * n_input_features, self.degree, 0)
        self._feature_indices = indices
        
        logger.info(f"多项式特征数: {len(indices)} (输入: {n_input_features})")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换为多项式特征
        Transform to polynomial features
        
        Args:
            X: 输入特征 (n_samples, n_features)
              Input features
        
        Returns:
            多项式特征 (n_samples, n_poly_features)
            Polynomial features
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        
        # 首次调用时自动fit
        # Auto-fit on first call
        if self._feature_indices is None:
            self.fit(X.shape[1])
        
        # 计算多项式特征
        # Compute polynomial features
        n_output_features = len(self._feature_indices)
        X_poly = np.ones((n_samples, n_output_features))
        
        for i, indices in enumerate(self._feature_indices):
            for sample_idx in range(n_samples):
                value = 1.0
                for feature_idx, power in enumerate(indices):
                    if power > 0:
                        value *= X[sample_idx, feature_idx] ** power
                X_poly[sample_idx, i] = value
        
        return X_poly
    
    def get_n_output_features(self, n_input_features: int) -> int:
        """
        获取输出特征数
        Get number of output features
        
        使用组合数公式
        Using combination formula:
        C(n+d, d) = (n+d)! / (n! * d!)
        
        Args:
            n_input_features: 输入特征数
                            Input feature count
        
        Returns:
            输出特征数
            Output feature count
        """
        from math import factorial
        n = n_input_features
        d = self.degree
        count = factorial(n + d) // (factorial(n) * factorial(d))
        if not self.include_bias:
            count -= 1
        return count


# ================================================================================
# 第9.5.2节：傅里叶基
# Section 9.5.2: Fourier Basis
# ================================================================================

class FourierBasis:
    """
    傅里叶基函数
    Fourier Basis Functions
    
    使用余弦函数的线性组合
    Linear combination of cosine functions
    
    φᵢ(s) = cos(πcᵢ·s)
    
    其中 where:
    - cᵢ: 频率向量
         Frequency vector
    - s: 归一化状态 [0,1]ᵈ
        Normalized state
    
    优势 Advantages:
    - 全局特征
      Global features
    - 适合周期函数
      Good for periodic functions
    - 快速学习
      Fast learning
    
    劣势 Disadvantages:
    - 需要归一化
      Requires normalization
    - 不适合不连续函数
      Not good for discontinuous functions
    """
    
    def __init__(self, n_features: int, bounds: List[Tuple[float, float]]):
        """
        初始化傅里叶基
        Initialize Fourier basis
        
        Args:
            n_features: 特征数（阶数）
                       Number of features (order)
            bounds: 每个维度的边界 [(min, max), ...]
                   Bounds for each dimension
        """
        self.n_features = n_features
        self.bounds = bounds
        self.n_dims = len(bounds)
        
        # 生成频率向量
        # Generate frequency vectors
        self.coefficients = self._generate_coefficients()
        
        logger.info(f"初始化傅里叶基: n={n_features}, dims={self.n_dims}")
    
    def _generate_coefficients(self) -> np.ndarray:
        """
        生成傅里叶系数
        Generate Fourier coefficients
        
        使用规则格子
        Using regular grid
        
        Returns:
            系数矩阵 (n_features, n_dims)
            Coefficient matrix
        """
        # 简化：使用前n个频率组合
        # Simplified: use first n frequency combinations
        coefficients = []
        
        # 计算每个维度的最大频率
        # Compute max frequency per dimension
        max_freq = int(np.ceil(self.n_features ** (1.0 / self.n_dims)))
        
        # 生成所有频率组合
        # Generate all frequency combinations
        for i in range(self.n_features):
            coef = []
            temp = i
            for d in range(self.n_dims):
                coef.append(temp % max_freq)
                temp //= max_freq
            coefficients.append(coef)
        
        return np.array(coefficients)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        归一化状态到[0,1]
        Normalize state to [0,1]
        
        Args:
            state: 原始状态
                  Raw state
        
        Returns:
            归一化状态
            Normalized state
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        normalized = np.zeros_like(state)
        for i, (low, high) in enumerate(self.bounds):
            if high > low:
                normalized[:, i] = (state[:, i] - low) / (high - low)
            else:
                normalized[:, i] = 0.5
        
        return np.clip(normalized, 0, 1)
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        转换为傅里叶特征
        Transform to Fourier features
        
        Args:
            state: 输入状态
                  Input state
        
        Returns:
            傅里叶特征
            Fourier features
        """
        # 归一化
        # Normalize
        normalized = self.normalize_state(state)
        
        if normalized.ndim == 1:
            normalized = normalized.reshape(1, -1)
        
        n_samples = normalized.shape[0]
        features = np.zeros((n_samples, self.n_features))
        
        # 计算余弦特征
        # Compute cosine features
        for i in range(self.n_features):
            dot_product = np.dot(normalized, self.coefficients[i])
            features[:, i] = np.cos(np.pi * dot_product)
        
        return features.squeeze() if n_samples == 1 else features


# ================================================================================
# 第9.5.3节：径向基函数
# Section 9.5.3: Radial Basis Functions
# ================================================================================

class RadialBasisFunction:
    """
    径向基函数 (RBF)
    Radial Basis Functions
    
    高斯核的线性组合
    Linear combination of Gaussian kernels
    
    φᵢ(s) = exp(-||s - cᵢ||² / (2σᵢ²))
    
    其中 where:
    - cᵢ: 中心点
         Center point
    - σᵢ: 宽度参数
         Width parameter
    
    特点 Characteristics:
    - 局部特征
      Local features
    - 适合局部泛化
      Good for local generalization
    - 可解释性好
      Good interpretability
    """
    
    def __init__(self, n_features: int, bounds: List[Tuple[float, float]],
                sigma: Optional[float] = None):
        """
        初始化RBF
        Initialize RBF
        
        Args:
            n_features: RBF中心数
                       Number of RBF centers
            bounds: 状态空间边界
                   State space bounds
            sigma: 高斯宽度（None则自动计算）
                  Gaussian width (auto if None)
        """
        self.n_features = n_features
        self.bounds = bounds
        self.n_dims = len(bounds)
        
        # 生成中心点
        # Generate centers
        self.centers = self._generate_centers()
        
        # 设置宽度
        # Set width
        if sigma is None:
            # 基于中心间距离
            # Based on inter-center distance
            distances = []
            for i in range(min(10, n_features)):
                for j in range(i+1, min(10, n_features)):
                    dist = np.linalg.norm(self.centers[i] - self.centers[j])
                    distances.append(dist)
            self.sigma = np.mean(distances) if distances else 1.0
        else:
            self.sigma = sigma
        
        logger.info(f"初始化RBF: n={n_features}, σ={self.sigma:.3f}")
    
    def _generate_centers(self) -> np.ndarray:
        """
        生成RBF中心
        Generate RBF centers
        
        使用均匀网格
        Using uniform grid
        
        Returns:
            中心点矩阵 (n_features, n_dims)
            Center matrix
        """
        # 计算每个维度的分割数
        # Compute splits per dimension
        n_per_dim = int(np.ceil(self.n_features ** (1.0 / self.n_dims)))
        
        # 生成网格点
        # Generate grid points
        centers = []
        for i in range(self.n_features):
            center = []
            temp = i
            for d in range(self.n_dims):
                low, high = self.bounds[d]
                idx = temp % n_per_dim
                temp //= n_per_dim
                
                if n_per_dim > 1:
                    value = low + (high - low) * idx / (n_per_dim - 1)
                else:
                    value = (low + high) / 2
                
                center.append(value)
            centers.append(center)
        
        return np.array(centers)
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        转换为RBF特征
        Transform to RBF features
        
        Args:
            state: 输入状态
                  Input state
        
        Returns:
            RBF特征
            RBF features
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        n_samples = state.shape[0]
        features = np.zeros((n_samples, self.n_features))
        
        # 计算到各中心的距离
        # Compute distances to centers
        for i in range(self.n_features):
            distances = np.linalg.norm(state - self.centers[i], axis=1)
            features[:, i] = np.exp(-distances**2 / (2 * self.sigma**2))
        
        return features.squeeze() if n_samples == 1 else features


# ================================================================================
# 第9.5.4节：瓦片编码
# Section 9.5.4: Tile Coding
# ================================================================================

class Iht:
    """
    索引哈希表 (IHT)
    Index Hash Table
    
    瓦片编码的哈希表实现
    Hash table implementation for tile coding
    
    避免存储所有可能的瓦片
    Avoid storing all possible tiles
    """
    
    def __init__(self, size: int):
        """
        初始化IHT
        Initialize IHT
        
        Args:
            size: 哈希表大小
                 Hash table size
        """
        self.size = size
        self.dict = {}
        self.count = 0
    
    def get_index(self, key: Tuple) -> int:
        """
        获取索引
        Get index
        
        Args:
            key: 瓦片键
                Tile key
        
        Returns:
            瓦片索引
            Tile index
        """
        if key not in self.dict:
            if self.count < self.size:
                self.dict[key] = self.count
                self.count += 1
            else:
                # 哈希冲突处理
                # Handle hash collision
                return hash(key) % self.size
        
        return self.dict[key]


class TileCoding:
    """
    瓦片编码
    Tile Coding
    
    Sutton的经典方法！
    Sutton's classic method!
    
    核心思想 Core Idea:
    - 多个偏移的网格（瓦片）
      Multiple offset grids (tilings)
    - 每个瓦片一个二进制特征
      One binary feature per tile
    - 稀疏表示
      Sparse representation
    
    优势 Advantages:
    - 计算效率极高
      Extremely efficient
    - 内存效率高
      Memory efficient
    - 可控的泛化
      Controllable generalization
    
    参数 Parameters:
    - n_tilings: 瓦片数
                Number of tilings
    - tile_width: 瓦片宽度
                 Tile width
    - iht_size: 哈希表大小
               Hash table size
    """
    
    def __init__(self, n_tilings: int, bounds: List[Tuple[float, float]],
                n_tiles_per_dim: int = 8, iht_size: int = 4096):
        """
        初始化瓦片编码
        Initialize tile coding
        
        Args:
            n_tilings: 瓦片数
                      Number of tilings
            bounds: 状态空间边界
                   State space bounds
            n_tiles_per_dim: 每维瓦片数
                            Tiles per dimension
            iht_size: IHT大小
                     IHT size
        """
        self.n_tilings = n_tilings
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.n_tiles_per_dim = n_tiles_per_dim
        
        # 索引哈希表
        # Index hash table
        self.iht = Iht(iht_size)
        
        # 计算瓦片宽度
        # Compute tile widths
        self.tile_widths = []
        for low, high in bounds:
            width = (high - low) / n_tiles_per_dim
            self.tile_widths.append(width)
        
        # 计算偏移
        # Compute offsets
        self.offsets = []
        for i in range(n_tilings):
            offset = []
            for j in range(self.n_dims):
                # 均匀偏移
                # Uniform offset
                offset.append(i * self.tile_widths[j] / n_tilings)
            self.offsets.append(offset)
        
        logger.info(f"初始化瓦片编码: tilings={n_tilings}, "
                   f"tiles/dim={n_tiles_per_dim}, iht={iht_size}")
    
    def get_tiles(self, state: np.ndarray) -> List[int]:
        """
        获取活跃瓦片
        Get active tiles
        
        Args:
            state: 状态
                  State
        
        Returns:
            活跃瓦片索引列表
            List of active tile indices
        """
        tiles = []
        
        for tiling_idx in range(self.n_tilings):
            # 计算瓦片坐标
            # Compute tile coordinates
            tile_coords = []
            for dim in range(self.n_dims):
                # 考虑偏移
                # Consider offset
                shifted = state[dim] + self.offsets[tiling_idx][dim]
                
                # 归一化到瓦片索引
                # Normalize to tile index
                low, high = self.bounds[dim]
                if high > low:
                    normalized = (shifted - low) / (high - low)
                    tile_idx = int(normalized * self.n_tiles_per_dim)
                    tile_idx = np.clip(tile_idx, 0, self.n_tiles_per_dim - 1)
                else:
                    tile_idx = 0
                
                tile_coords.append(tile_idx)
            
            # 生成瓦片键
            # Generate tile key
            tile_key = (tiling_idx,) + tuple(tile_coords)
            
            # 获取瓦片索引
            # Get tile index
            tile_index = self.iht.get_index(tile_key)
            tiles.append(tile_index)
        
        return tiles
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        转换为稀疏二进制特征
        Transform to sparse binary features
        
        Args:
            state: 输入状态
                  Input state
        
        Returns:
            稀疏特征向量
            Sparse feature vector
        """
        # 获取活跃瓦片
        # Get active tiles
        active_tiles = self.get_tiles(state)
        
        # 创建稀疏特征
        # Create sparse features
        features = np.zeros(self.iht.size)
        for tile_idx in active_tiles:
            features[tile_idx] = 1.0
        
        return features
    
    def get_n_active_features(self) -> int:
        """
        获取活跃特征数
        Get number of active features
        
        Returns:
            每个状态的活跃特征数
            Active features per state
        """
        return self.n_tilings


# ================================================================================
# 主函数：演示特征构造
# Main Function: Demonstrate Feature Construction
# ================================================================================

def demonstrate_feature_construction():
    """
    演示特征构造方法
    Demonstrate feature construction methods
    """
    print("\n" + "="*80)
    print("第9.5节：特征构造")
    print("Section 9.5: Feature Construction")
    print("="*80)
    
    # 创建示例2D状态
    # Create example 2D state
    bounds = [(0, 10), (0, 10)]
    test_states = np.array([
        [2.5, 7.5],
        [5.0, 5.0],
        [7.5, 2.5]
    ])
    
    print(f"\n测试状态空间: {bounds}")
    print(f"测试状态:")
    for i, state in enumerate(test_states):
        print(f"  s{i+1}: {state}")
    
    # 1. 多项式特征
    # 1. Polynomial Features
    print("\n" + "="*60)
    print("1. 多项式特征")
    print("1. Polynomial Features")
    print("="*60)
    
    poly = PolynomialFeatures(degree=2, include_bias=True)
    
    for i, state in enumerate(test_states):
        features = poly.transform(state)
        print(f"\ns{i+1} -> 多项式特征 (dim={len(features)}):")
        print(f"  前5个特征: {features[:5]}")
        print(f"  特征范数: {np.linalg.norm(features):.3f}")
    
    # 2. 傅里叶基
    # 2. Fourier Basis
    print("\n" + "="*60)
    print("2. 傅里叶基")
    print("2. Fourier Basis")
    print("="*60)
    
    fourier = FourierBasis(n_features=16, bounds=bounds)
    
    for i, state in enumerate(test_states):
        features = fourier.transform(state)
        print(f"\ns{i+1} -> 傅里叶特征 (dim={len(features)}):")
        print(f"  前5个特征: {features[:5]}")
        print(f"  特征范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # 3. 径向基函数
    # 3. Radial Basis Functions
    print("\n" + "="*60)
    print("3. 径向基函数 (RBF)")
    print("3. Radial Basis Functions")
    print("="*60)
    
    rbf = RadialBasisFunction(n_features=9, bounds=bounds)
    
    print(f"\nRBF中心点:")
    for i in range(min(5, len(rbf.centers))):
        print(f"  c{i+1}: {rbf.centers[i]}")
    print(f"  σ = {rbf.sigma:.3f}")
    
    for i, state in enumerate(test_states):
        features = rbf.transform(state)
        print(f"\ns{i+1} -> RBF特征 (dim={len(features)}):")
        print(f"  前5个特征: {features[:5]}")
        print(f"  活跃特征数 (>0.1): {np.sum(features > 0.1)}")
    
    # 4. 瓦片编码
    # 4. Tile Coding
    print("\n" + "="*60)
    print("4. 瓦片编码")
    print("4. Tile Coding")
    print("="*60)
    
    tiles = TileCoding(n_tilings=8, bounds=bounds, 
                      n_tiles_per_dim=4, iht_size=512)
    
    for i, state in enumerate(test_states):
        active_tiles = tiles.get_tiles(state)
        features = tiles.transform(state)
        
        print(f"\ns{i+1} -> 瓦片编码:")
        print(f"  活跃瓦片: {active_tiles}")
        print(f"  特征维度: {len(features)}")
        print(f"  活跃特征数: {np.sum(features > 0)}")
        print(f"  稀疏度: {1 - np.sum(features > 0) / len(features):.1%}")
    
    # 5. 比较不同方法
    # 5. Compare different methods
    print("\n" + "="*60)
    print("5. 方法比较")
    print("5. Method Comparison")
    print("="*60)
    
    print(f"\n{'方法':<20} {'特征数':<15} {'类型':<15} {'特点':<30}")
    print("-" * 80)
    
    print(f"{'多项式 (d=2)':<20} {6:<15} {'全局':<15} "
          f"{'光滑，维度爆炸':<30}")
    print(f"{'傅里叶 (n=16)':<20} {16:<15} {'全局':<15} "
          f"{'周期性，需归一化':<30}")
    print(f"{'RBF (n=9)':<20} {9:<15} {'局部':<15} "
          f"{'局部泛化，可解释':<30}")
    print(f"{'瓦片编码 (8x4x4)':<20} {512:<15} {'局部':<15} "
          f"{'稀疏高效，内存友好':<30}")
    
    # 测试泛化
    # Test generalization
    print("\n" + "="*60)
    print("6. 泛化测试")
    print("6. Generalization Test")
    print("="*60)
    
    # 测试相近状态的特征相似度
    # Test feature similarity for nearby states
    state1 = np.array([5.0, 5.0])
    state2 = np.array([5.1, 5.1])  # 相近状态
    state3 = np.array([8.0, 2.0])  # 远离状态
    
    print(f"\n基准状态: {state1}")
    print(f"相近状态: {state2} (距离: {np.linalg.norm(state2-state1):.2f})")
    print(f"远离状态: {state3} (距离: {np.linalg.norm(state3-state1):.2f})")
    
    # 计算特征余弦相似度
    # Compute feature cosine similarity
    def cosine_similarity(f1, f2):
        return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
    
    print(f"\n特征相似度 (余弦):")
    print(f"{'方法':<20} {'相近状态':<15} {'远离状态':<15}")
    print("-" * 50)
    
    # 多项式
    f1_poly = poly.transform(state1)
    f2_poly = poly.transform(state2)
    f3_poly = poly.transform(state3)
    print(f"{'多项式':<20} {cosine_similarity(f1_poly, f2_poly):<15.3f} "
          f"{cosine_similarity(f1_poly, f3_poly):<15.3f}")
    
    # 傅里叶
    f1_fourier = fourier.transform(state1)
    f2_fourier = fourier.transform(state2)
    f3_fourier = fourier.transform(state3)
    print(f"{'傅里叶':<20} {cosine_similarity(f1_fourier, f2_fourier):<15.3f} "
          f"{cosine_similarity(f1_fourier, f3_fourier):<15.3f}")
    
    # RBF
    f1_rbf = rbf.transform(state1)
    f2_rbf = rbf.transform(state2)
    f3_rbf = rbf.transform(state3)
    print(f"{'RBF':<20} {cosine_similarity(f1_rbf, f2_rbf):<15.3f} "
          f"{cosine_similarity(f1_rbf, f3_rbf):<15.3f}")
    
    # 瓦片
    f1_tiles = tiles.transform(state1)
    f2_tiles = tiles.transform(state2)
    f3_tiles = tiles.transform(state3)
    print(f"{'瓦片编码':<20} {cosine_similarity(f1_tiles, f2_tiles):<15.3f} "
          f"{cosine_similarity(f1_tiles, f3_tiles):<15.3f}")
    
    # 总结
    # Summary
    print("\n" + "="*80)
    print("特征构造总结")
    print("Feature Construction Summary")
    print("="*80)
    
    print("""
    关键要点 Key Takeaways:
    =======================
    
    1. 特征决定泛化模式
       Features determine generalization pattern
       
    2. 全局vs局部特征
       Global vs local features
       
    3. 多项式：光滑但维度爆炸
       Polynomial: Smooth but dimension explosion
       
    4. 傅里叶：周期性函数
       Fourier: Periodic functions
       
    5. RBF：局部泛化
       RBF: Local generalization
       
    6. 瓦片编码：高效稀疏
       Tile coding: Efficient and sparse
    
    选择建议 Selection Tips:
    - 低维光滑：多项式
      Low-dim smooth: Polynomial
    - 周期性：傅里叶
      Periodic: Fourier
    - 局部相关：RBF
      Local correlation: RBF
    - 大规模：瓦片编码
      Large-scale: Tile coding
    """)


if __name__ == "__main__":
    demonstrate_feature_construction()