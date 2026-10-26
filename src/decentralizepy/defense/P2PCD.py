"""P2PCD: two-norm clipping defense for P2P federated learning.
Syros et al., "Backdoor Attacks in Peer-to-Peer Federated Learning", 2024, Section 6.3 ("Our Defense").
"""

import logging
import torch


class P2PCD:
    def __init__(self, clip_neigh=0.1, clip_local=1.0, agreement_rounds=0):
        """
        Parameters
        ----------
        clip_neigh : float
            l2 clipping norm for updates received from neighbors (0.1 in paper)
        clip_local : float
            l2 clipping norm for the node's own post-training update (1.0).
        agreement_rounds : int
            Initial agreeement phase
        """
        self.clip_neigh = float(clip_neigh)
        self.clip_local = float(clip_local)
        self.agreement_rounds = int(agreement_rounds)
        self.theta_prev = None

    @staticmethod
    def _snapshot(state_dict):
        return {
            k: v.detach().clone()
            for k, v in state_dict.items()
            if torch.is_floating_point(v)
        }

    def snapshot_pre_round(self, model):
        """Record theta_prev at the start of round t (pre-training, pre-agg).
        """
        self.theta_prev = self._snapshot(model.state_dict())

    def _global_l2_norm(self, update):
        """l2 norm over the concatenation of all update tensors (paper eq. 5)."""
        total = 0.0
        for t in update.values():
            total += float(t.detach().pow(2).sum().item())
        return total ** 0.5

    def _clip_toward_prev(self, target_state, clip_norm):
        """Rescale target_state so that ||target_state - theta_prev||_2 <= clip_norm.
        """
        if self.theta_prev is None:
            return target_state, 1.0

        update = {}
        for key, val in target_state.items():
            ref = self.theta_prev.get(key)
            if ref is None or not torch.is_floating_point(val) or val.shape != ref.shape:
                continue
            update[key] = val - ref.to(val.device)

        if not update:
            return target_state, 1.0

        norm = self._global_l2_norm(update)
        scale = 1.0
        if norm > clip_norm and norm > 0.0:
            scale = clip_norm / norm
            for key, u in update.items():
                ref = self.theta_prev[key].to(target_state[key].device)
                target_state[key] = ref + u * scale

        return target_state, scale

    def should_clip(self, iteration):
        """True once the agreement phase is over and clipping should be enforced."""
        return iteration >= self.agreement_rounds and self.theta_prev is not None

    def clip_peer_state(self, peer_state_dict, iteration):
        """Clip an incoming neighbor model toward theta_prev with C_neigh.
        """
        if not self.should_clip(iteration):
            return peer_state_dict, 1.0
        state, scale = self._clip_toward_prev(peer_state_dict, self.clip_neigh)
        if scale < 1.0:
            logging.debug("P2PCD: neighbor update scaled by %.4f (C=%.3f)",
                          scale, self.clip_neigh)
        return state, scale

    def clip_local_model(self, model, iteration):
        """Clip the locally trained model toward theta_prev with C_local.
        """
        if not self.should_clip(iteration):
            return 1.0
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        state_dict, scale = self._clip_toward_prev(state_dict, self.clip_local)
        if scale < 1.0:
            model.load_state_dict(state_dict)
            logging.debug("P2PCD: local update scaled by %.4f (C=%.3f)",
                          scale, self.clip_local)
        return scale
