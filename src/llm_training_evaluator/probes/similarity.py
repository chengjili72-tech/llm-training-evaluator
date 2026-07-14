from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..adapters.base import ModelAdapter


class TokenSimilarityProbe:
    def __init__(self, adapter: ModelAdapter, tokenizer: Any, chunk_size: int = 16_384) -> None:
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def analyze(self, input_ids: Tensor, top_k: int = 10) -> list[dict[str, object]]:
        embedding_module = self.adapter.input_embeddings()
        weight = getattr(embedding_module, "weight", None)
        if weight is None or not isinstance(weight, Tensor):
            raise TypeError("Input embedding module does not expose a tensor weight")

        unique_ids = list(dict.fromkeys(int(value) for value in input_ids.flatten().tolist()))
        query_ids = torch.tensor(unique_ids, device=weight.device, dtype=torch.long)
        queries = torch.nn.functional.normalize(weight[query_ids].float(), dim=-1)
        effective_k = min(top_k, max(weight.shape[0] - 1, 1))
        best_values = torch.full(
            (len(unique_ids), effective_k), -torch.inf, device=weight.device
        )
        best_ids = torch.full(
            (len(unique_ids), effective_k), -1, device=weight.device, dtype=torch.long
        )
        excluded = set(getattr(self.tokenizer, "all_special_ids", []))

        for start in range(0, weight.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, weight.shape[0])
            candidates = torch.nn.functional.normalize(weight[start:stop].float(), dim=-1)
            similarities = queries @ candidates.T
            candidate_ids = torch.arange(start, stop, device=weight.device)

            for query_row, query_id in enumerate(unique_ids):
                mask_ids = excluded | {query_id}
                local = [token_id - start for token_id in mask_ids if start <= token_id < stop]
                if local:
                    similarities[query_row, local] = -torch.inf

            expanded_ids = candidate_ids.unsqueeze(0).expand(len(unique_ids), -1)
            merged_values = torch.cat((best_values, similarities), dim=-1)
            merged_ids = torch.cat((best_ids, expanded_ids), dim=-1)
            best_values, selected = torch.topk(merged_values, k=effective_k, dim=-1)
            best_ids = torch.gather(merged_ids, dim=-1, index=selected)

        rows: list[dict[str, object]] = []
        for row, token_id in enumerate(unique_ids):
            neighbors = []
            for rank, (neighbor_id, similarity) in enumerate(
                zip(best_ids[row].tolist(), best_values[row].tolist()), start=1
            ):
                neighbors.append(
                    {
                        "token_id": int(neighbor_id),
                        "token": str(self.tokenizer.convert_ids_to_tokens(int(neighbor_id))),
                        "text": self.tokenizer.decode(
                            [int(neighbor_id)], clean_up_tokenization_spaces=False
                        ),
                        "similarity": float(similarity),
                        "rank": rank,
                    }
                )
            rows.append(
                {
                    "token_id": token_id,
                    "token": str(self.tokenizer.convert_ids_to_tokens(token_id)),
                    "text": self.tokenizer.decode(
                        [token_id], clean_up_tokenization_spaces=False
                    ),
                    "neighbors": neighbors,
                }
            )
        return rows
