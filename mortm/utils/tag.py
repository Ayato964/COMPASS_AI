from typing import Dict, List, Union
import numpy as np
import torch
from torch import Tensor


def extract_tagged_sequences_batch(
    batch_tensor: Union[Tensor, np.ndarray, list],
    start_tag_id: int,
    end_tag_id: int,
    include_tags: bool = True,
) -> List[List[Tensor]]:
    """
    Extracts sequences enclosed by start and end tags from a batch of tensors.
    Optimized to minimize GPU-CPU synchronization points.
    """
    if isinstance(batch_tensor, (np.ndarray, list)):
        batch_tensor = torch.as_tensor(batch_tensor)
    if batch_tensor.dim() == 1:
        batch_tensor = batch_tensor.unsqueeze(0)
    batch_size = batch_tensor.shape[0]
    results: List[List[Tensor]] = [[] for _ in range(batch_size)]

    start_indices = (batch_tensor == start_tag_id).nonzero(as_tuple=False)
    end_indices = (batch_tensor == end_tag_id).nonzero(as_tuple=False)

    starts_list = start_indices.tolist()
    ends_list = end_indices.tolist()

    starts_by_batch: Dict[int, List[int]] = {b: [] for b in range(batch_size)}
    ends_by_batch: Dict[int, List[int]] = {b: [] for b in range(batch_size)}

    for b_idx, s_idx in starts_list:
        starts_by_batch[b_idx].append(s_idx)

    for b_idx, s_idx in ends_list:
        ends_by_batch[b_idx].append(s_idx)

    for b in range(batch_size):
        s_indices = starts_by_batch[b]
        e_indices = ends_by_batch[b]

        if not s_indices or not e_indices:
            continue

        last_end_pos = -1
        e_search_start = 0
        num_ends = len(e_indices)

        for s_pos in s_indices:
            if s_pos <= last_end_pos:
                continue

            found_e_pos = -1
            for k in range(e_search_start, num_ends):
                e_pos = e_indices[k]
                if e_pos > s_pos:
                    found_e_pos = e_pos
                    e_search_start = k + 1
                    break

            if found_e_pos != -1:
                if include_tags:
                    segment = batch_tensor[b, s_pos : found_e_pos + 1]
                else:
                    segment = batch_tensor[b, s_pos + 1 : found_e_pos]
                results[b].append(segment)
                last_end_pos = found_e_pos

    return results
