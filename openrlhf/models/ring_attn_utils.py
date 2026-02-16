import torch
import torch.distributed as dist
import torch.nn.functional as F

def index_first_axis(input, indices):
    return input[indices]

def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    output = torch.zeros(batch * seqlen, dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, dim)

def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    output = hidden_states.flatten(0, 1)[indices]
    
    return output, indices, cu_seqlens, max_seqlen_in_batch, None 

def rearrange(x, *args, **kwargs):
    try:
        import einops
        return einops.rearrange(x, *args, **kwargs)
    except ImportError:
        raise ImportError("Please install einops: pip install einops")

def all_gather(tensor, group=None):
    if group is None:
        return tensor
    
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return tensor
        
    tensors_gather = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, group=group)
    output = torch.cat(tensors_gather, dim=0)
    return output

RING_ATTN_GROUP = None

def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group

def get_ring_attn_group():
    return RING_ATTN_GROUP

def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids

def update_ring_attn_params(cu_seqlens):
    if RING_ATTN_GROUP is not None:
         print("Warning: update_ring_attn_params called but ring_flash_attn is missing.")
         pass

def get_tensor_in_current_ring_attn_rank(tensors: list[torch.Tensor] | torch.Tensor, ring_attn_group, pad_id):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    
    if ring_attn_group is None:
        return tensors if isinstance(tensors, list) else [tensors], 0

    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    ring_attn_pad_len = (ring_attn_size - seqlen % ring_attn_size) % ring_attn_size
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, ring_attn_pad_len), value=pad_id)
        local_seq_len = tensor.numel() // ring_attn_size
        start, end = ring_attn_rank * local_seq_len, (ring_attn_rank + 1) * local_seq_len
        tensor = tensor[:, start:end]
        output_tensors.append(tensor)
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]
    return output_tensors, ring_attn_pad_len


def unpad_and_slice_tensor(sequences, attention_mask, ring_attn_group):
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    
    sequences = sequences.transpose(0, 1)
    
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(
        0, 1
    ) 
    
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )
    
    ring_attn_pad_len = 0
    if ring_attn_group is not None:
        (sequences, position_ids, rolled_sequences), ring_attn_pad_len = get_tensor_in_current_ring_attn_rank(
            [sequences, position_ids, rolled_sequences], ring_attn_group, 0
        )
        cu_seqlens[-1] += ring_attn_pad_len
        update_ring_attn_params(cu_seqlens)
        
    return sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices


def gather_and_pad_tensor(tensor, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen):
    if ring_attn_group is not None:
        tensor = all_gather(tensor.transpose(0, 1), ring_attn_group).transpose(0, 1)
        if ring_attn_pad_len > 0:
            tensor = tensor[:, :-ring_attn_pad_len]
            
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)
    return tensor