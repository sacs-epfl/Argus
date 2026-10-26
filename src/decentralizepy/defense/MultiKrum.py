import logging

import torch


class MultiKrumDefense:
    """
    Multi-Krum adapted for decentralized learning.

    Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", 2017.

    Parameters
    ----------
    num_to_reject : int
        Number of neighbour models to drop per round (default 1).
    """

    def __init__(self, num_to_reject=1):
        self.num_to_reject = num_to_reject

    def select(self, neighbor_models):
        """
        Given a mapping {neighbor_id: state_dict}, return the set of neighbor_ids to keep.

        Parameters
        ----------
        neighbor_models : dict[int, dict[str, Tensor]]

        Returns: set[int]
        """
        ids = list(neighbor_models.keys())
        n = len(ids)

        if n == 0:
            return set()

        num_reject = min(self.num_to_reject, n - 1) # always keep at least 1
        if num_reject <= 0:
            return set(ids)

        # Flatten each model into a 1D float vector
        vectors = {
            nid: torch.cat([v.detach().float().reshape(-1) for v in state.values()])
            for nid, state in neighbor_models.items()
        }

        # Pairwise squared Euclidean distances
        dist = {}
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i < j:
                    d = (vectors[id_i] - vectors[id_j]).norm().item()**2
                    dist[(id_i, id_j)] = d
                    dist[(id_j, id_i)] = d

        # Krum score for each model
        k = max(1, n - num_reject - 1)
        scores = {}
        for nid in ids:
            dists_to_others = sorted(dist[(nid, other)] for other in ids if other != nid)
            scores[nid] = sum(dists_to_others[:k])

        # Reject the num_reject models with the highest scores
        sorted_ids = sorted(ids, key=lambda x: scores[x])
        rejected = set(sorted_ids[-num_reject:])
        selected = set(ids) - rejected

        logging.info(
            "Multi-Krum: kept %d/%d neighbours (num_to_reject=%d): "
            "rejected=%s",
            len(selected), n, num_reject, rejected,
        )

        return selected
