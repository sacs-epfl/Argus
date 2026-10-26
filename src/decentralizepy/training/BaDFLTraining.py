"""Training wrapper that applies the BaDFL local defense around each round.
"""

from decentralizepy.defense.BaDFL import BaDFL
from decentralizepy.training.Training import Training


class BaDFLTraining(Training):
    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        badfl_alpha=1.0,
        badfl_clip_q=0.1,
    ):
        super().__init__(
            rank,
            machine_id,
            mapping,
            model,
            optimizer,
            loss,
            log_dir,
            rounds=rounds,
            full_epochs=full_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.badfl = BaDFL(alpha=float(badfl_alpha), clip_q=float(badfl_clip_q))

    def train(self, dataset):
        # Malicious nodes skip the defense entirely.
        is_malicious = bool(getattr(dataset, "malicious", False))
        if is_malicious:
            super().train(dataset)
            return
        self.badfl.before_train(self.model)
        super().train(dataset)
        gamma = self.optimizer.param_groups[0]["lr"]
        self.badfl.after_train(self.model, gamma)
