from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    # 1. 基底周波数の計算
    # head_dimは偶数を想定。0,2,4...と2つずつペアを作る
    frequencies = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (frequencies / head_dim))
    
    # 2. 各位置(m)での角度を計算
    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)  # (seqlen, head_dim/2)
    
    # 3. cos/sinを計算
    freqs_cos = torch.cos(freqs)  # (seqlen, head_dim/2)
    freqs_sin = torch.sin(freqs)  # (seqlen, head_dim/2)
    
    # 4. クエリとキーを実部と虚部に分割
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    
    # 5. ブロードキャストのための形状調整
    # freqs_cos: (seqlen, head_dim/2) -> (1, seqlen, 1, head_dim/2)
    freqs_cos = freqs_cos.view(1, seqlen, 1, freqs_cos.shape[-1])
    freqs_sin = freqs_sin.view(1, seqlen, 1, freqs_sin.shape[-1])
    
    # 6. 回転の適用
    # スライドの式を実装:
    # 実部: x_real * cos - x_imag * sin
    # 虚部: x_real * sin + x_imag * cos
    query_out_real = query_real * freqs_cos - query_imag * freqs_sin
    query_out_imag = query_real * freqs_sin + query_imag * freqs_cos
    key_out_real = key_real * freqs_cos - key_imag * freqs_sin
    key_out_imag = key_real * freqs_sin + key_imag * freqs_cos
    
    # 7. 実部と虚部を結合して元の形状に戻す
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1)
    query_out = query_out.reshape(query.shape).type_as(query)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1)
    key_out = key_out.reshape(key.shape).type_as(key)
    return query_out, key_out