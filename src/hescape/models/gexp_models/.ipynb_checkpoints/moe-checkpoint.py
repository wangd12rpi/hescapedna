import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_fwd, custom_bwd


# ---------- 按专家分块的并行Linear Layer（一层线性层） ----------
# MoE路由后，会把本批次中要送到每个专家的样本块按顺序拼在一起，形成一个大矩阵inputs
# 同时又一个长度为E的张量 expert_size=[n_0, n_1, ... n_{E-1}]，表示每个专家拿了多少条样本
# ParallelLinear就是不显式for-loop出专家维度的情况下完成分块矩阵乘，然后把所有结果再顺序拼接回一个大输出矩阵
# 反向传播同理按块求梯度并顺序拼接回去
class ParallelLinear(torch.autograd.Function):
    '''
    expert_size: [E], 每个专家的样本数
    input: [sum(expert_size), in_dim]: sum(expert_size)表示所有专家的样本数之和
    weight: [E, in_dim, out_dim], 每个专家一套权重
    bias: [E, out_dim] / None
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx,
                input: Tensor,
                expert_size: Tensor,
                weight: Tensor,
                bias: Optional[Tensor]
                ) -> Tensor:
        output = ParallelLinear.forward_scriptable(
            input, expert_size, weight, bias)
        ctx.save_for_backward(input, expert_size, weight, bias)
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(input: Tensor,
                           expert_size: Tensor,
                           weight: Tensor,
                           bias: Optional[Tensor]) -> Tensor:
        # 预分配输出缓冲
        out_dim = weight.size(2)
        output_buf: Tensor = torch.empty((input.size(0), out_dim),
                                         device=input.device, dtype=input.dtype)
        num_experts: int = weight.size(0)

        expert_size_list: List[int] = expert_size.tolist()
        # 按 expert_size 切分输入与输出缓冲，避免 for 循环里切片拷贝
        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list, dim=0)

        # 前向：对每个专家做一次 mm
        for i in range(num_experts):
            # output_i = input_i @ W_i
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_experts):
                output_buf_list[i].add_(bias[i])  # 行加 bias_i

        return output_buf

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out: Tensor):
        input, expert_size, weight, bias = ctx.saved_tensors
        return ParallelLinear.backward_scriptable(grad_out, input, expert_size, weight, bias)

    @staticmethod
    @torch.jit.script
    def backward_scriptable(grad_out: Tensor,
                            input: Tensor,
                            expert_size: Tensor,
                            weight: Tensor,
                            bias: Optional[Tensor]
                            ) -> Tuple[Tensor, None, Tensor, Optional[Tensor]]:
        num_experts: int = weight.size(0)
        expert_size_list: List[int] = expert_size.tolist()

        # 切分
        input_t_list = input.t().split(
            expert_size_list, dim=1)          # [in_dim, n_i]
        grad_list = grad_out.split(
            expert_size_list, dim=0)              # [n_i, out_dim]

        # 反向 dInput
        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(
            expert_size_list, dim=0)    # [n_i, in_dim]
        # [E, out_dim, in_dim]
        weight_t = weight.permute(0, 2, 1)
        for i in range(num_experts):
            # dInput_i = dY_i @ W_i^T
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])

        # 反向 dWeight
        # [E, in_dim, out_dim]
        d_weight_buf = torch.empty_like(weight)
        for i in range(num_experts):
            # dW_i = X_i^T @ dY_i
            torch.mm(input_t_list[i], grad_list[i], out=d_weight_buf[i])

        # 反向 dBias
        if bias is not None:
            d_bias_buf = torch.empty_like(
                bias)                           # [E, out_dim]
            for i in range(num_experts):
                torch.sum(grad_list[i], dim=0,
                          keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        # expert_size 无需梯度
        return d_input_buf, None, d_weight_buf, d_bias


# ---------- 把 top-k 路由结果组织成 按专家聚合的索引 + 门控权重，同时统计每个专家的负载 ----------
@torch.jit.script
def compute_gating(k: int,
                   probs: Tensor,            # [N, E]（仅用于统计 expert_size/gates）
                   top_k_gates: Tensor,      # [N, k]
                   top_k_indices: Tensor     # [N, k]
                   ):
    '''
    k: 每个token被分配到的专家个数
    N: batch_size * sequence_length (展平后的token数)
    probs: [N, E], 每个token对所有专家的概率分布
    top_k_gates: [N, K], 每个token选出的top-k概率值
    top_k_indices: [N, K], 每个token选出的对应的专家id
    '''
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)   # [N, E] 稀疏门控（统计用）

    top_k_gates_flat = top_k_gates.flatten()               # [N*k]
    top_k_experts_flat = top_k_indices.flatten()           # [N*k]

    # 仅保留非零门控
    nonzeros = top_k_gates_flat.nonzero().squeeze(-1)      # mask 索引
    top_k_experts_nz = top_k_experts_flat[nonzeros]        # 对应专家 id

    # 将 token-expert 对按专家排序，得到重排索引
    _, order = top_k_experts_nz.sort(0)
    index_sorted_experts = nonzeros[order]                  # [M]，M<=N*k

    # expert_size：每个专家的 token 频次（>0）
    expert_size = (gates > 0).long().sum(0)                # [E]

    # batch_index：回写时使用（原 token 索引 = 平展索引 // k）
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')  # [M]
    batch_gates = top_k_gates_flat[index_sorted_experts]              # [M]

    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


# ---------- 把专家的线性层打包到一起，并通过 ParallelLinear 实现并行前向/反向计算 ----------
class ParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_experts, input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(
            num_experts, output_size)) if bias else None
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'num_experts={self.weight.size(0)}, input_size={self.weight.size(1)}, output_size={self.weight.size(2)}'

    def reset_parameters(self) -> None:
        # 简单的均匀初始化
        nn.init.uniform_(self.weight, -1.0 / self.weight.size(1),
                         1.0 / self.weight.size(1))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor, expert_size: Tensor) -> Tensor:
        # inputs: [sum(expert_size), in_dim]
        return ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)


# ---------- 稀疏MoE层 ----------
class MoE(nn.Module):
    """
    input_size: 输入/输出特征维度
    head_size:  专家隐藏维
    num_experts: 专家个数 E
    k: 每个 token 选择的专家数
    cvloss, switchloss, zloss: 对应辅助损失系数
    bias: 专家线性层是否带偏置
    activation: 专家 FFN 的激活函数, 默认 GELU
    noisy_gating: 是否使用 Noisy Top-k
    acc_aux_loss: 是否跨 step 累积统计后再取一次辅助损失
    """

    def __init__(self,
                 input_size: int,
                 head_size: int,
                 num_experts: int,
                 k: int,
                 cvloss: float = 0.0,
                 switchloss: float = 0.0,
                 zloss: float = 0.0,
                 bias: bool = False,
                 gating_activation: Optional[nn.Module] = None,  # 预留
                 activation: Optional[nn.Module] = None,
                 noisy_gating: bool = True,
                 acc_aux_loss: bool = False):
        super().__init__()

        assert num_experts >= 1, "num_experts must be >= 1"
        assert 1 <= k <= num_experts, "k must be in [1, num_experts]"

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.k = k

        self.experts = ParallelExperts(
            num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(
            num_experts, head_size, input_size, bias)

        self.cvloss_coef = cvloss
        self.switchloss_coef = switchloss
        self.zloss_coef = zloss
        self.activation = activation or nn.GELU()

        # 门控网络：线性映射
        out_dim = 2 * num_experts if noisy_gating else num_experts
        self.f_gate = nn.Linear(input_size, out_dim, bias=False)
        nn.init.zeros_(self.f_gate.weight)  # 均匀初始路由；若关闭 noisy 可考虑小随机初始化

        # 跨 step 累积的统计量
        self.acc_aux_loss = acc_aux_loss
        if acc_aux_loss:
            self.register_buffer("acc_probs", torch.zeros(num_experts))
            self.register_buffer("acc_gates", torch.zeros(num_experts))
            self.register_buffer("acc_freq", torch.zeros(num_experts))
            self.register_buffer("acc_lsesq", torch.tensor(0.0))
            self.register_buffer("acc_lsesq_count", torch.tensor(0.0))

        # forward 后会缓存的路由索引/尺寸
        self.expert_size: Optional[Tensor] = None
        self.index_sorted_experts: Optional[Tensor] = None
        self.batch_index: Optional[Tensor] = None
        self.batch_gates: Optional[Tensor] = None

    def extra_repr(self) -> str:
        return f'k={self.k}, cvloss={self.cvloss_coef}, switchloss={self.switchloss_coef}, zloss={self.zloss_coef}, noisy_gating={self.noisy_gating}'

    # --------- 辅助损失（单步计算） ----------
    @staticmethod
    def cv_squared(x: Tensor) -> Tensor:
        """CV^2 = Var(x) / (Mean(x)^2 + eps)，鼓励更均匀的专家使用。"""
        eps = 1e-10
        if x.numel() == 0 or x.shape[0] == 1:
            return x.new_tensor(0.0)
        x = x.float()
        return x.var(unbiased=False) / (x.mean() ** 2 + eps)

    @staticmethod
    def _zloss_from_logits(logits: Tensor) -> Tensor:
        return torch.mean(torch.logsumexp(logits, dim=1) ** 2)

    def compute_cvloss(self, probs_or_gates: Tensor) -> Tensor:
        per_expert = probs_or_gates.sum(0)              # [E]
        per_expert = F.normalize(per_expert, p=1, dim=0)
        return self.cv_squared(per_expert)

    def compute_switchloss(self, probs: Tensor, freqs: Tensor) -> Tensor:
        p = F.normalize(probs.sum(0), p=1, dim=0)
        q = F.normalize(freqs.float(), p=1, dim=0)
        return (p * q).sum() * self.num_experts

    def compute_zloss(self, logits: Tensor) -> Tensor:
        return self._zloss_from_logits(logits)

    # --------- 累积统计 ----------
    def init_aux_statistics(self):
        # 清零累积统计
        assert self.acc_aux_loss, "acc_aux_loss=False 时无需调用"
        self.acc_probs.zero_()
        self.acc_gates.zero_()
        self.acc_freq.zero_()
        self.acc_lsesq.zero_()
        self.acc_lsesq_count.zero_()

    def update_aux_statistics(self, logits: Tensor, probs: Tensor, gates: Tensor):
        lsesq = torch.logsumexp(logits, dim=1) ** 2
        self.acc_probs += probs.sum(0).detach()
        self.acc_gates += gates.sum(0).detach()
        self.acc_freq += (gates > 0).float().sum(0).detach()
        self.acc_lsesq += lsesq.sum().detach()
        self.acc_lsesq_count += torch.tensor(
            float(lsesq.size(0)), device=lsesq.device)

    def get_aux_loss_and_clear(self) -> Tensor:
        """取一次聚合辅助损失并清空累积"""
        assert self.acc_aux_loss, "acc_aux_loss=False 时无需调用"
        eps = 1e-10
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count + eps)
        loss = (self.cvloss_coef * cvloss +
                self.switchloss_coef * switchloss +
                self.zloss_coef * zloss)
        self.init_aux_statistics()
        return loss

    # --------- Noisy Top-k 路由 ----------
    def top_k_gating(self,
                     x: Tensor,                  # [N, d]
                     # [N, 1]，为 True 的 token 跳过
                     skip_mask: Optional[Tensor] = None,
                     sample_topk: int = 0,       # 可选：部分采样 top-k（默认禁用）
                     noise_epsilon: float = 1e-2) -> Tensor:
        """
        本 step 的辅助损失（若 acc_aux_loss=False 直接返回标量；否则返回 0 并把统计累积起来）
        并在模块上缓存以下成员供 forward/reduce 使用：
          - self.expert_size, self.index_sorted_experts, self.batch_index, self.batch_gates
        """
        clean_out = self.f_gate(
            x)                             # [N, 2E] 或 [N, E]
        if self.noisy_gating:
            clean_logits, raw_noise_std = clean_out.chunk(
                2, dim=-1)  # [N, E], [N, E]
            noise_std = F.softplus(raw_noise_std) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_out

        if skip_mask is not None:
            logits = logits.masked_fill(skip_mask, float('-inf'))

        probs = torch.softmax(logits, dim=1)                    # [N, E]

        # 选择 top-k
        if self.training and (sample_topk > 0):
            assert 0 < sample_topk <= self.k
            # 先取 k-sample_topk 个最大，再在剩余概率里多项式采样 sample_topk 个
            top_km1_gates, top_km1_idx = probs.topk(
                self.k - sample_topk, dim=1)
            masked = probs + 1e-6
            masked[torch.arange(probs.size(0)).unsqueeze(1), top_km1_idx] = 0
            k_idx_sample = torch.multinomial(masked, sample_topk)
            top_k_indices = torch.cat([top_km1_idx, k_idx_sample], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)   # [N, k]

        # 计算路由索引与统计
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # 缓存给 forward/reduce 使用
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        # 辅助损失
        aux_loss = x.new_tensor(0.0)
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates)
        else:
            aux_loss = (self.cvloss_coef * self.compute_cvloss(gates) +
                        self.switchloss_coef * self.compute_switchloss(probs, self.expert_size) +
                        self.zloss_coef * self.compute_zloss(logits))

        self._last_topk_indices = top_k_indices    # [B*L, K]
        self._last_topk_gates = top_k_gates      # [B*L, K]

        return aux_loss

    # --------- 前向：完整 MoE（map -> 激活 -> reduce） ----------
    def forward(self,
                x: Tensor,                              # [B, L, d]
                # [B, L, 1] 或 [B, L] -> True 表示跳过
                skip_mask: Optional[Tensor] = None,
                sample_topk: int = 0,
                multiply_by_gates: bool = True
                ) -> Tuple[Tensor, Tensor]:
        B, L, D = x.size()
        assert D == self.input_size

        x_flat = x.reshape(-1, D)                       # [N, d]
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1).bool()    # [N, 1]

        aux_loss = self.top_k_gating(x_flat, skip_mask, sample_topk)

        assign_flat = self._last_topk_indices[:, 0]           # [B*L]
        self.last_expert_assign = assign_flat.view(B, L)[:, 0].detach()

        # 收集路由后的 token
        assert self.batch_index is not None and self.expert_size is not None and self.index_sorted_experts is not None
        # [M, d]，M<=N*k（去掉 gate=0 的项）
        expert_inputs = x_flat[self.batch_index]

        # 专家第一段线性 + 激活
        h = self.experts(expert_inputs, self.expert_size)   # [M, head]
        h = self.activation(h)

        # 专家第二段线性
        expert_outputs = self.output_experts(h, self.expert_size)  # [M, d]

        # 乘以对应 gate
        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        # 汇聚回原 token 位置：index_add 以 token 索引聚合 k 路
        y_flat = torch.zeros(
            (B * L, D), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y_flat.index_add_(0, self.batch_index, expert_outputs)
        y = y_flat.view(B, L, D)
        return y, aux_loss

    # --------- map/reduce 拆分 ----------
    def map(self,
            x: Tensor,                              # [B, L, d]
            skip_mask: Optional[Tensor] = None,
            sample_topk: int = 0
            ) -> Tuple[Tensor, Tensor]:
        """仅做路由 + 专家第一段线性，返回 [B, L, k, head]"""
        B, L, D = x.size()
        x_flat = x.reshape(-1, D)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1).bool()

        aux_loss = self.top_k_gating(x_flat, skip_mask, sample_topk)
        assert self.batch_index is not None and self.expert_size is not None and self.index_sorted_experts is not None

        expert_inputs = x_flat[self.batch_index]                # [M, d]
        expert_out1 = self.experts(
            expert_inputs, self.expert_size)  # [M, head]

        # 将专家输出按 index_sorted_experts 写回到 [N*k, head] 的容器，然后 reshape 为 [B, L, k, head]
        y_buf = torch.zeros((B * L * self.k, self.head_size),
                            dtype=expert_out1.dtype, device=expert_out1.device)
        y_buf.index_add_(0, self.index_sorted_experts, expert_out1)
        y = y_buf.view(B, L, self.k, self.head_size)
        return y, aux_loss

    def reduce(self,
               x_k: Tensor,                           # [B, L, k, head]
               multiply_by_gates: bool = True
               ) -> Tensor:
        """接 map 的输出继续做第二段线性并聚合回原维度"""
        B, L, k, H = x_k.size()
        assert k == self.k and H == self.head_size
        # [N*k, head]
        x_flat = x_k.view(-1, H)

        assert self.index_sorted_experts is not None and self.batch_index is not None and self.expert_size is not None
        # [M, head]
        expert_inputs = x_flat[self.index_sorted_experts]
        expert_out2 = self.output_experts(
            expert_inputs, self.expert_size)  # [M, d]

        if multiply_by_gates:
            expert_out2 = expert_out2 * self.batch_gates[:, None]

        y_flat = torch.zeros((B * L, self.input_size),
                             dtype=expert_out2.dtype, device=expert_out2.device)
        y_flat.index_add_(0, self.batch_index, expert_out2)
        return y_flat.view(B, L, self.input_size)


# if __name__ == "__main__":
#     torch.manual_seed(0)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     B, L, D = 2, 4, 16
#     H = 32
#     E = 4
#     K = 2

#     moe = MoE(input_size=D, head_size=H, num_experts=E, k=K,
#               cvloss=0.01, switchloss=0.01, zloss=1e-4,
#               bias=False, noisy_gating=True, acc_aux_loss=False).to(device)

#     x = torch.randn(B, L, D, device=device)
#     y, aux = moe(x)

#     print("y:", y.shape, "aux_loss:", float(aux))
#     print("expert_size:", moe.expert_size.cpu().numpy())   # 每个 expert 分到多少 token
