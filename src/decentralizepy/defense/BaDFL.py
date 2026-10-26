"""BaDFL: Mitigating Model Poisoning in Decentralized Federated Learning. Yuan et al., 2025.
"""

import logging

import torch


class BaDFL:
    def __init__(self, alpha=1.0, clip_q=0.1):
        """
        Parameters
        ----------
        alpha : float
            Strength of the regularization step (eq. 10).
        clip_q : float
            Elementwise clipping threshold for G (eq. 9).
        """
        self.alpha = float(alpha)
        self.clip_q = float(clip_q)
        self.x_prev = None # post-aggregation snapshot from round t-1
        self.x_curr = None # post-aggregation snapshot from round t

    @staticmethod
    def _snapshot(model):
        return {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    def before_train(self, model):
        """Record X_t (post-aggregation, pre-training) at the start of round t."""
        self.x_curr = self._snapshot(model)

    def after_train(self, model, gamma):
        """
        Apply eqs. (8)-(10) in place on the post-training weights.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose post-training weights X_tilde_{t+1} should be corrected.
        gamma : float
            Local learning rate gamma_t used for the SGD step that produced X_tilde.
        """
        x_tilde = self._snapshot(model)

        if self.x_prev is None or self.x_curr is None:
            # Need two consecutive post-aggregation snapshots before Delta is defined. Shift history and skip the correction this round.
            self.x_prev = self.x_curr
            self.x_curr = None
            logging.debug("BaDFL: warm-up round, correction skipped")
            return

        state_dict = model.state_dict()

        total_abs_g = 0.0
        total_abs_delta = 0.0
        total_params = 0
        for key, x_t_tilde in x_tilde.items():
            x_t = self.x_curr.get(key)
            x_tm1 = self.x_prev.get(key)
            if x_t is None or x_tm1 is None:
                continue
            if x_t_tilde.shape != x_t.shape or x_t_tilde.shape != x_tm1.shape:
                continue

            delta = x_t - x_tm1
            gamma_h = (x_t_tilde - x_t) - delta # eq. (8): gamma_t * H_tilde
            g = (1.0 - gamma_h).clamp(-self.clip_q, self.clip_q) # eq. (9): clip(I - gamma_t * H_tilde, q)
            corrected = x_t_tilde - gamma * self.alpha * g # eq. (10)
            state_dict[key].copy_(corrected)

            total_abs_g += float(g.abs().sum().item())
            total_abs_delta += float(delta.abs().sum().item())
            total_params += g.numel()

        model.load_state_dict(state_dict)

        if total_params > 0:
            logging.debug(
                "BaDFL: alpha=%.3f q=%.4f mean|G|=%.6f mean|Delta|=%.6f",
                self.alpha,
                self.clip_q,
                total_abs_g / total_params,
                total_abs_delta / total_params,
            )

        self.x_prev = self.x_curr
        self.x_curr = None
