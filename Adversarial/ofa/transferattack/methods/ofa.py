import torch
import torch.nn as nn
from ..attack import Attack
from ..utils import diverse_input_transform, generate_ti_kernel, ti_smooth_grad


class OFA(Attack):
    """
    Orthogonal Feature Attack (OFA)
    
    核心创新：
    1. 正交子空间投影：在ResNet主成分的正交补空间中随机采样mask
       - 避免ResNet bias（主成分 = 模型特定偏好）
       - 保留特征空间几何结构（不是盲目随机）
    2. ILPD特征衰减：混合对抗特征和干净特征，保证梯度对齐
    3. 梯度聚合：K次不同正交mask，消除噪声，增强信号
    
    理论基础：
    设ResNet的前k个主成分为 V = [v1, ..., vk]
    正交补空间 V⊥ = {x ∈ R^C : x⊥vi, ∀i≤k}
    
    在V⊥中扰动：
    - ResNet对V⊥中的方向不敏感（正交性）
    - ViT可能对V⊥中的方向敏感（架构差异）
    - 实现高迁移性：对源模型温和，对目标模型强烈
    
    Arguments:
        model_name (str): 源模型名称
        epsilon (float): 扰动预算，默认16/255
        alpha (float): 步长，默认1.6/255
        epoch (int): 迭代次数，默认300
        decay (float): momentum衰减因子，默认1.0
        targeted (bool): 是否为目标攻击，默认False
        random_start (bool): 是否随机初始化，默认False
        norm (str): 范数类型，默认'linfty'
        loss (str): 损失函数，默认'crossentropy'
        device (torch.device): 设备，默认None
        attack (str): 攻击名称，默认'OFA'
        
        # OFA核心参数
        K (int): 梯度聚合次数，默认5
        gamma (float): ILPD特征衰减因子，默认2
        enable_feature_decay (bool): 是否启用ILPD特征衰减，默认True
        n_components (int): PCA主成分数量，默认32
        layer_names (list): 操作的层名称，默认['layer1', 'layer3']
        
        # TI/DI参数
        enable_ti (bool): 是否启用Translation Invariant，默认False
        kernel_type (str): TI核类型，默认'gaussian'
        kernel_size (int): TI核大小，默认5
        enable_di (bool): 是否启用Diverse Input，默认False
        di_scale_factor (float): DI缩放因子，默认1.14
        
    Example script:
        python main.py --attack ofa --model=resnet50 --disable_ti --disable_di
        python main.py --attack ofa --model=resnet50 --eval --disable_ti --disable_di
    """
    
    def __init__(
        self,
        model_name,
        epsilon=16/255,
        alpha=1/255,
        epoch=100,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='OFA',
        # OFA核心参数
        K=5,  # 梯度聚合次数，控制正交mask采样数量，K越大迁移性越好但计算开销增加
        gamma=2,
        enable_feature_decay=True,  # 控制是否启用ILPD特征衰减机制
        n_components=32,
        layer_names=['layer1', 'layer3'],
        # TI/DI参数
        enable_ti=False,
        kernel_type='gaussian',
        kernel_size=5,
        enable_di=False,
        di_scale_factor=1.14,
        **kwargs,
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

        self.enable_ti = enable_ti
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        if self.enable_ti:
            self.kernel = generate_ti_kernel(self.kernel_type, self.kernel_size, self.device)

        self.enable_di = enable_di
        self.di_scale_factor = di_scale_factor
        
        # OFA核心参数
        self.K = K
        self.gamma = gamma
        self.enable_feature_decay = enable_feature_decay
        self.n_components = n_components
        self.layer_names = layer_names
        
        # 缓存
        self._pca_components = {}  # {layer_name: [k, C]} 每层的主成分
        self._clean_features = {}  # {layer_name: [B, C, H, W]} 干净特征
        self.hook_handles = []
        
        print(f"[OFA] Initialized with:")
        print(f"  K={self.K}, gamma={self.gamma}, feature_decay={'ON' if self.enable_feature_decay else 'OFF'}, n_components={self.n_components}")
        print(f"  layers={self.layer_names}")
        print(f"  TI={'ON' if self.enable_ti else 'OFF'}, DI={'ON' if self.enable_di else 'OFF'}")
    
    def forward(self, data, label, **kwargs):
        """
        OFA主攻击循环
        
        流程：
        1. 预计算：提取干净特征 + 计算PCA主成分
        2. 注册hook：在目标层应用正交mask + 特征衰减
        3. 迭代优化：聚合梯度 + 更新momentum + 更新delta
        4. 清理：移除hook并返回
        
        Args:
            data: 原始图像 [B, C, H, W]
            label: 标签 [B] 或 [2, B]（如果targeted）
        
        Returns:
            delta: 对抗扰动 [B, C, H, W]
        """
        # 处理targeted attack的label
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # 使用目标标签
        
        # 移动到device
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Step 1: 预计算干净特征和PCA主成分
        self._precompute(data)
        
        # Step 2: 注册hook
        self._register_hooks()
        
        # Step 3: 初始化delta和momentum
        delta = self.init_delta(data)
        momentum = 0
        
        # Step 4: 主循环
        for iter_idx in range(self.epoch):
            # 【核心】计算聚合梯度
            aggregated_grad = self._compute_aggregated_gradient(
                data, delta, label
            )
            
            # 更新momentum
            momentum = self.get_momentum(aggregated_grad, momentum)
            
            # 更新delta
            delta = self.update_delta(delta, data, momentum, self.alpha)
        
        # Step 5: 清理
        self._remove_hooks()
        self._clean_features.clear()
        self._pca_components.clear()
        
        return delta.detach()
    
    def _precompute(self, data):
        """
        预计算阶段（只执行一次）
        
        目的：
        1. 提取干净图像的特征（用于ILPD特征衰减）
        2. 计算PCA主成分（用于定义正交补空间）
        
        Args:
            data: 干净图像 [B, C, H, W]
        """
        with torch.no_grad():
            # 临时hook用于提取特征
            features = {}
            
            def make_feature_hook(name):
                def hook_fn(module, input, output):
                    features[name] = output.detach().clone()
                return hook_fn
            
            # 注册临时hook
            temp_handles = []
            for layer_name in self.layer_names:
                layer = self._find_layer(self.model, layer_name)
                if layer is not None:
                    handle = layer.register_forward_hook(make_feature_hook(layer_name))
                    temp_handles.append(handle)
                else:
                    raise ValueError(f"[OFA] Layer '{layer_name}' not found in model")
            
            # Forward获取干净特征
            _ = self.model(data)
            
            # 移除临时hook
            for handle in temp_handles:
                handle.remove()
            
            # 处理每一层
            for layer_name, feat in features.items():
                # 保存干净特征（用于特征衰减）
                self._clean_features[layer_name] = feat
                
                # 计算PCA主成分（用于定义正交空间）
                components, _, _ = self._compute_pca(feat, self.n_components)
                self._pca_components[layer_name] = components  # [k, C]
                
                print(f"[OFA] Precomputed layer={layer_name}, "
                      f"feature_shape={feat.shape}, components_shape={components.shape}")
    
    def _compute_pca(self, features, n_components):
        """
        计算PCA分解
        
        Args:
            features: 特征图 [B, C, H, W]
            n_components: 主成分数量
        
        Returns:
            components: [n_components, C] 主成分向量（已归一化）
            mean: [C] 均值
            explained_variance: [n_components] 方差解释比例
        """
        B, C, H, W = features.shape
        n_components = min(n_components, C)
        
        # 展平空间维度：[B, C, H, W] -> [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # 中心化
        mean = features_flat.mean(dim=0)  # [C]
        centered = features_flat - mean.unsqueeze(0)  # [B*H*W, C]
        
        # 数值稳定性：添加小噪声避免退化
        std = centered.std(dim=0)
        if (std < 1e-6).any():
            centered = centered + torch.randn_like(centered) * 1e-6
        
        # SVD分解：X = U S V^T
        try:
            U, S, Vt = torch.linalg.svd(centered.T, full_matrices=False)
            # U: [C, min(C, B*H*W)]
            # S: [min(C, B*H*W)]
            # Vt: [min(C, B*H*W), B*H*W]
        except RuntimeError as e:
            print(f"[OFA] SVD failed: {e}, using fallback")
            # Fallback: 使用单位矩阵
            U = torch.eye(C, device=features.device)
            S = torch.ones(min(C, centered.shape[0]), device=features.device)
        
        # 主成分向量（已归一化）
        components = U[:, :n_components].T  # [n_components, C]
        
        # 方差解释比例
        total_var = (S ** 2).sum()
        explained_variance = (S[:n_components] ** 2) / (total_var + 1e-8)
        
        return components, mean, explained_variance
    
    def _compute_aggregated_gradient(self, data, delta, label):
        """
        OFA核心：通过K次正交mask聚合梯度
        
        理论：
        每次随机mask在正交补空间V⊥中采样，产生的梯度：
            g_k = g_universal + g_model_specific_k
        
        其中：
        - g_universal: 跨模型一致的方向（对正交mask鲁棒）
        - g_model_specific_k: 模型特定方向（对正交mask敏感）
        
        聚合后：
            G = (1/K) Σ g_k ≈ g_universal
        
        Args:
            data: 原始图像 [B, C, H, W]
            delta: 当前扰动 [B, C, H, W]
            label: 标签 [B]
        
        Returns:
            aggregated_grad: 聚合后的梯度 [B, C, H, W]
        """
        # 初始化累积梯度
        aggregated_grad = torch.zeros_like(delta)
        
        for k in range(self.K):
            # Step 1: 创建独立的x_adv（避免计算图冲突）
            x_adv = (data + delta).detach().requires_grad_(True)
            
            # Step 2: 应用DI变换（如果启用）
            x_transformed = self.transform(x_adv)
            
            # Step 3: Forward（hook自动应用正交mask + 特征衰减）
            # 注意：每次forward，_apply_orthogonal_mask会重新采样正交mask
            logits_k = self.get_logits(x_transformed)
            
            # Step 4: 计算loss
            loss_k = self.get_loss(logits_k, label)
            
            # Step 5: 计算梯度
            grad_k = torch.autograd.grad(
                loss_k, 
                x_adv,
                retain_graph=(k < self.K - 1),  # 前K-1次保留图
                create_graph=False
            )[0]
            
            # Step 6: 累积梯度
            aggregated_grad = aggregated_grad + grad_k
        
        # Step 7: 平均
        aggregated_grad = aggregated_grad / self.K
        
        # Step 8: 应用TI平滑（如果启用）
        if self.enable_ti and hasattr(self, 'kernel'):
            aggregated_grad = ti_smooth_grad(
                aggregated_grad, 
                self.kernel, 
                enable_ti=True
            )
        
        return aggregated_grad
    
    def _apply_orthogonal_mask(self, features, layer_name):
        """
        OFA核心：应用正交空间mask + ILPD特征衰减
        
        步骤：
        1. 生成随机mask logits（连续值）
        2. 投影到主成分的正交补空间V⊥
        3. 转换为binary mask（伯努利采样）
        4. 应用mask（保持期望能量）
        5. ILPD特征衰减（混合干净特征）
        
        Args:
            features: 特征图 [B, C, H, W]
            layer_name: 层名称
        
        Returns:
            decayed_features: 处理后的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Step 1: 生成随机mask logits（连续值，未归一化）
        mask_logits = torch.randn(B, C, 1, 1, device=features.device)
        
        # Step 2: 投影到正交补空间V⊥
        components = self._pca_components[layer_name]  # [k, C]
        mask_orthogonal = self._orthogonal_projection(mask_logits, components)
        
        # Step 3: 转换为binary mask
        # 使用sigmoid将连续值映射到[0, 1]，然后伯努利采样
        mask_prob = torch.sigmoid(mask_orthogonal)
        mask = torch.bernoulli(mask_prob)  # [B, C, 1, 1]
        
        # Step 4: 应用mask（保持期望能量不变）
        # E[features * mask / E[mask]] = features
        mask_mean = mask.mean() + 1e-8
        masked_features = features * mask / mask_mean
        
        # Step 5: ILPD特征衰减（可选）
        if self.enable_feature_decay:
            clean_features = self._clean_features[layer_name]
            decayed_features = self._feature_decay(
                masked_features, clean_features, self.gamma
            )
        else:
            decayed_features = masked_features
        
        return decayed_features
    
    def _orthogonal_projection(self, mask_logits, components):
        """
        投影到主成分的正交补空间
        
        数学公式：
            P_V⊥(x) = x - P_V(x)
            P_V(x) = V(V^T V)^{-1} V^T x
        
        简化（假设V已正交化）：
            P_V⊥(x) = x - V V^T x
        
        Args:
            mask_logits: 随机mask logits [B, C, 1, 1]
            components: PCA主成分 [k, C]（已归一化）
        
        Returns:
            mask_orthogonal: 正交投影后的mask [B, C, 1, 1]
        """
        B, C, _, _ = mask_logits.shape
        k = components.shape[0]
        
        # 展平mask: [B, C, 1, 1] -> [B, C]
        mask_flat = mask_logits.squeeze(-1).squeeze(-1)  # [B, C]
        
        # 计算投影系数：coeff = V^T x
        # [B, C] @ [C, k] = [B, k]
        coeff = torch.matmul(mask_flat, components.T)  # [B, k]
        
        # 投影到主成分空间：proj_V = V coeff = V (V^T x)
        # [B, k] @ [k, C] = [B, C]
        proj_V = torch.matmul(coeff, components)  # [B, C]
        
        # 正交投影：P_V⊥(x) = x - proj_V
        mask_orthogonal = mask_flat - proj_V  # [B, C]
        
        # 恢复形状：[B, C] -> [B, C, 1, 1]
        return mask_orthogonal.unsqueeze(-1).unsqueeze(-1)
    
    def _feature_decay(self, f_adv, f_clean, gamma):
        """
        ILPD特征衰减机制
        
        数学公式：
            f̃ = (1/γ) f_adv + (1 - 1/γ) f_clean
        
        作用：
        - 防止过大扰动导致梯度方向偏差
        - 强制对抗特征保持在干净特征附近
        - 提升梯度对齐度 → 增强迁移性
        
        Args:
            f_adv: 对抗特征（mask后） [B, C, H, W]
            f_clean: 干净特征 [B, C, H, W]
            gamma: 衰减因子（越大衰减越强）
        
        Returns:
            f_decayed: 衰减后的特征 [B, C, H, W]
        """
        decay_factor = 1.0 / gamma
        return decay_factor * f_adv + (1.0 - decay_factor) * f_clean
    
    def _register_hooks(self):
        """
        注册前向传播hook（多层）
        
        每个hook会在对应层的forward后，自动调用_apply_orthogonal_mask
        """
        self.hook_handles = []
        
        for layer_name in self.layer_names:
            # 创建闭包捕获layer_name
            def make_hook_fn(name):
                def hook_fn(module, input, output):
                    return self._apply_orthogonal_mask(output, name)
                return hook_fn
            
            # 查找目标层
            layer = self._find_layer(self.model, layer_name)
            if layer is not None:
                handle = layer.register_forward_hook(make_hook_fn(layer_name))
                self.hook_handles.append(handle)
                print(f"[OFA] Registered hook on {layer_name}")
            else:
                print(f"[OFA] Warning: Layer '{layer_name}' not found, skipping")
    
    def _remove_hooks(self):
        """移除所有hook"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        print(f"[OFA] All hooks removed")
    
    def _find_layer(self, model, layer_name):
        """
        查找目标层
        
        Args:
            model: 模型
            layer_name: 层名称（如'layer1', 'layer3'）
        
        Returns:
            module: 找到的层，如果未找到则返回None
        """
        # 如果model是Sequential包装的，提取实际模型
        if isinstance(model, nn.Sequential):
            model = model[-1]
        
        # 遍历所有命名模块
        for name, module in model.named_modules():
            if name == layer_name or name.endswith('.' + layer_name):
                return module
        
        return None
    
    def get_logits(self, x, **kwargs):
        """
        前向传播
        
        注意：这个方法会触发hook，从而调用_apply_orthogonal_mask
        """
        return self.model(x)

    def transform(self, data, **kwargs):
        """Apply DI (Diverse Input) transform when enabled."""
        return diverse_input_transform(
            data,
            di_scale_factor=self.di_scale_factor,
            enable_di=self.enable_di,
        )
