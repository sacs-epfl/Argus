import importlib
import ast
import json
import logging
import math
import os
from collections import deque

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node

def _energy_map(trigger):
    """To (H, W) energy map."""
    if trigger.dim() == 3:
        return trigger.abs().mean(dim=0)
    return trigger.abs()


def _clip_topk(trigger, k):
    """
    Zero out all but the top-k pixels (by energy) in a trigger tensor.
    """
    energy = _energy_map(trigger) # (H, W)
    flat = energy.flatten()
    if flat.numel() <= k:
        return trigger.clone()
    threshold = torch.topk(flat, k).values[-1]
    mask = (energy >= threshold).float() # (H, W)
    if trigger.dim() == 3:
        return trigger * mask.unsqueeze(0)
    return trigger * mask


def default_trigger_similarity(trigger1, trigger2, k=49, window_size=11, threshold=0.35):
    """
    SSIM-based trigger similarity with top-k energy clipping.

    Steps:
      1) Clip each trigger to its top-k highest-energy pixels (for null distributionto be independent of mask density).
      2) Compute the mean local SSIM between the two clipped energy maps using a (window_size x window_size) sliding window.
    """
    if trigger1 is None or trigger2 is None:
        return False

    t1 = _clip_topk(trigger1, k)
    t2 = _clip_topk(trigger2, k)

    e1 = _energy_map(t1).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    e2 = _energy_map(t2).unsqueeze(0).unsqueeze(0)

    # Dynamic-range-calibrated SSIM constants (Wang et al., 2004)
    L = max(e1.max().item(), e2.max().item())
    L = max(L, 1e-8)  # avoid division by zero for all-zero maps
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=trigger1.device) / (window_size ** 2)

    mu1 = F.conv2d(e1, kernel, padding=pad)
    mu2 = F.conv2d(e2, kernel, padding=pad)
    sigma1_sq = F.conv2d(e1 ** 2, kernel, padding=pad) - mu1 ** 2
    sigma2_sq = F.conv2d(e2 ** 2, kernel, padding=pad) - mu2 ** 2
    sigma12   = F.conv2d(e1 * e2, kernel, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    score = ssim_map.mean().item()
    logging.debug(f"Trigger similarity (SSIM, k={k}): {score:.4f} (threshold={threshold})")
    return score >= threshold

class DPSGDNode(Node):
    """
    This class defines the node for DPSGD

    """

    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def get_neighbors(self, node=None):
        return self.my_neighbors

    def receive_DPSGD(self):
        return self.receive_channel("DPSGD")
    
    def receive_trigger_share(self, block=True):
        return self.receive_channel("TRIGGER_SHARE", block=block)

    def receive_trigger_barrier(self):
        return self.receive_channel("TRIGGER_BARRIER")
    
    def get_common_neighbors(self, suspected_node):
        """
        Get all neighbors of a suspected node (excluding ourselves).
        """
        suspected_neighbors = set(self.graph.neighbors(suspected_node))
        suspected_neighbors.discard(self.uid)
        return suspected_neighbors

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration

            # Dynamic malicious behaviour
            if self.dataset.malicious:
                # Activation: start malicious behavior
                if (
                    not self.malicious_activated
                    and iteration >= self.malicious_start_round
                    # Don't re-activate via start_round if we already stopped
                    and not (self.malicious_stop_round >= 0 and iteration >= self.malicious_stop_round)
                ):
                    self.malicious_activated = True
                    if hasattr(self.dataset, 'set_poisoning_active'):
                        self.dataset.set_poisoning_active(True)
                    self.malicious_ignore_averaging = self._malicious_ignore_averaging_target
                    logging.info(
                        ">>> Malicious activation at round %d: malicious behavior enabled "
                        "(poisoning=True, ignore_averaging=%s)",
                        iteration, self.malicious_ignore_averaging,
                    )

                # Deactivation: stop malicious behavior
                elif (
                    self.malicious_activated
                    and self.malicious_stop_round >= 0
                    and iteration >= self.malicious_stop_round
                    # Don't deactivate if we already restarted
                    and not (self.malicious_restart_round >= 0 and iteration >= self.malicious_restart_round)
                ):
                    self.malicious_activated = False
                    if hasattr(self.dataset, 'set_poisoning_active'):
                        self.dataset.set_poisoning_active(False)
                    self.malicious_ignore_averaging = False
                    logging.info(
                        ">>> Malicious deactivation at round %d: reverting to normal behavior "
                        "(poisoning=False, ignore_averaging=False)",
                        iteration,
                    )

                # Restart: resume malicious behavior
                elif (
                    not self.malicious_activated
                    and self.malicious_restart_round >= 0
                    and iteration >= self.malicious_restart_round
                ):
                    self.malicious_activated = True
                    if hasattr(self.dataset, 'set_poisoning_active'):
                        self.dataset.set_poisoning_active(True)
                    self.malicious_ignore_averaging = self._malicious_ignore_averaging_target
                    logging.info(
                        ">>> Malicious restart at round %d: malicious behavior re-enabled "
                        "(poisoning=True, ignore_averaging=%s)",
                        iteration, self.malicious_ignore_averaging,
                    )

            is_local_pretrain = iteration < self.local_pretrain_rounds

            if is_local_pretrain:
                if iteration == 0:
                    logging.info(
                        "Local pretraining phase: %d rounds before decentralized learning begins",
                        self.local_pretrain_rounds,
                    )
                logging.info(
                    "Local pretraining iteration %d/%d (training only, ignoring neighbor models)",
                    iteration + 1, self.local_pretrain_rounds,
                )
            elif iteration == self.local_pretrain_rounds and self.local_pretrain_rounds > 0:
                logging.info(
                    "Local pretraining complete after %d rounds. "
                    "Starting decentralized learning.",
                    self.local_pretrain_rounds,
                )

            # P2PCD
            if (
                getattr(self, "p2pcd", None) is not None
                and not self.dataset.malicious
                and not is_local_pretrain
            ):
                self.p2pcd.snapshot_pre_round(self.model)

            self.trainer.train(self.dataset)

            new_neighbors = self.get_neighbors()

            # The following code does not work because TCP sockets are supposed to be long lived.
            # for neighbor in self.my_neighbors:
            #     if neighbor not in new_neighbors:
            #         logging.info("Removing neighbor {}".format(neighbor))
            #         if neighbor in self.peer_deques:
            #             assert len(self.peer_deques[neighbor]) == 0
            #             del self.peer_deques[neighbor]
            #         self.communication.destroy_connection(neighbor, linger = 10000)
            #         self.barrier.remove(neighbor)

            self.my_neighbors = new_neighbors
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                self._drain_queued_trigger_requests(max_messages=32)
                sender, data = self.receive_DPSGD()
                logging.debug(
                    "Received Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
                )
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            averaging_deque = dict()
            for neighbor in self.my_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            if is_local_pretrain:
                # During local pretraining: discard received models, keep local model
                for n in averaging_deque:
                    averaging_deque[n].popleft()
                self.sharing.communication_round += 1
            elif self.malicious_ignore_averaging:
                # Malicious nodes can choose to ignore received updates
                for n in averaging_deque:
                    averaging_deque[n].popleft()
                self.sharing.communication_round += 1
                # Participate in the collaborative protocol adversarially
                if self.collaborative_defense:
                    self._trigger_barrier()
                    self._malicious_collaborative_participate()
            else:
                # Oracle defense: benign nodes drop models from known malicious neighbors
                if self._should_ignore_known_malicious():
                    for n in list(averaging_deque.keys()):
                        if n in self.malicious_node_ids:
                            averaging_deque[n].popleft()
                            del averaging_deque[n]
                            logging.debug(
                                "Oracle: dropped model from known malicious node %d "
                                "(round %d)", n, self.iteration,
                            )

                # Apply Local defense if enabled (non-malicious nodes only)
                if hasattr(self, 'defense_mode') and (self.defense is not None or self.defense_mode == 'oracle'):
                    filtered_deque = self.apply_defense(averaging_deque)
                    if filtered_deque is not None:
                        self.sharing._averaging(filtered_deque)
                elif self.malicious_activated and self.malicious_self_weight_boost != 1.0:
                    self.sharing._averaging(
                        averaging_deque,
                        self_weight_boost=self.malicious_self_weight_boost,
                    )
                else:
                    self.sharing._averaging(averaging_deque)

            # Post-averaging backdoor re-injection for malicious nodes
            if (
                self.malicious_activated
                and self.malicious_retrain_poison_rounds > 0
                and not is_local_pretrain
                and not self.malicious_ignore_averaging
            ):
                self.model.train()
                retrain_loader = self.dataset.get_trainset(
                    self.trainer.batch_size, self.trainer.shuffle,
                )
                retrain_count = 0
                retrain_loss = 0.0
                for data, target in retrain_loader:
                    retrain_loss += self.trainer.trainstep(data, target)
                    retrain_count += 1
                    if retrain_count >= self.malicious_retrain_poison_rounds:
                        break
                logging.info(
                    "Post-averaging re-injection: %d steps, avg loss %.4f",
                    retrain_count,
                    retrain_loss / max(retrain_count, 1),
                )

            if not is_local_pretrain and self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if self.scheduler is not None and not is_local_pretrain:
                self.scheduler.step()
                logging.debug("LR after step: %.6f", self.scheduler.get_last_lr()[0])

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "validation_loss": {},
                    "validation_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "backdoor_asr": {},
                    "backdoor_clean_acc": {},
                    "backdoor_trig_acc": {},
                    "activation_score": {},
                    "defense_rejected": {},
                    "defense_accepted": {},
                    "defense_rejection_rate": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes
            
            # Record defense statistics
            if hasattr(self, 'defense_stats'):
                results_dict["defense_rejected"][iteration + 1] = self.defense_stats["rejected"]
                results_dict["defense_accepted"][iteration + 1] = self.defense_stats["accepted"]
                total = self.defense_stats["rejected"] + self.defense_stats["accepted"]
                if total > 0:
                    results_dict["defense_rejection_rate"][iteration + 1] = self.defense_stats["rejected"] / total
                else:
                    results_dict["defense_rejection_rate"][iteration + 1] = 0.0

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl
                
                # Backdoor monitoring
                try:
                    asr, clean_acc, trig_acc = self.dataset.test_backdoor(self.model, num_samples=1000, require_clean_non_target_pred=self.require_clean_non_target_pred)
                    results_dict["backdoor_asr"][iteration + 1] = asr
                    results_dict["backdoor_clean_acc"][iteration + 1] = clean_acc
                    results_dict["backdoor_trig_acc"][iteration + 1] = trig_acc
                    logging.info(f"Backdoor ASR: {asr:.1f}% | Clean: {clean_acc:.1f}% | Triggered: {trig_acc:.1f}%")
                    
                    # Compute activation score if enabled
                    if self.activation_tracking_enabled:
                        try:
                            from decentralizepy.utils import compute_activation_score
                            activation_score = compute_activation_score(
                                self.model,
                                self.dataset,
                                num_samples=self.activation_tracking_samples
                            )
                            results_dict["activation_score"][iteration + 1] = activation_score
                            logging.info(f"Activation Score: {activation_score:.2f}")
                        except Exception as e:
                            logging.warning(f"Failed to compute activation score: {e}")
                except Exception as e:
                    logging.debug(f"Backdoor test skipped: {e}")
                
                if self.dataset.__validating__:
                    logging.info("Evaluating on the validation set")
                    va, vl = self.dataset.validate(self.model, self.loss)
                    results_dict["validation_acc"][iteration + 1] = va
                    results_dict["validation_loss"][iteration + 1] = vl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.cpu().numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, self.iterations - 1)
        logging.info("All neighbors disconnected. Process complete!")

        if self.rank == 0:
            import time
            time.sleep(90)  # Wait for all nodes to finish
            logging.info("Evaluating average model on test set (excluding malicious nodes)...")
            data = dict()
            for p_rank in range(self.mapping.get_n_procs()):
                p_uid = self.mapping.get_uid(p_rank, self.machine_id)
                if p_uid in self.malicious_node_ids:
                    logging.info("Skipping malicious node %d for final average model.", p_uid)
                    continue
                weight_path = os.path.join(
                    self.weights_store_dir, f"{self.iterations - 1}_weight_{p_uid}.pt"
                )
                
                data[p_uid] = torch.load(weight_path, map_location="cuda:0")
            # Average all models into self.model
            with torch.no_grad():
                for key in self.model.state_dict().keys():
                    avg_param = torch.zeros_like(self.model.state_dict()[key], dtype=torch.float32)
                    for uid in data:
                        avg_param += data[uid][key]
                    avg_param /= len(data)
                    self.model.state_dict()[key].copy_(avg_param) 
            ta, tl = self.dataset.test(self.model, self.loss)
            logging.info("Average Model Test Accuracy: %.2f%%", ta)
            logging.info("Average Model Test Loss: %.4f", tl)
            try:
                asr, clean_acc, trig_acc = self.dataset.test_backdoor(self.model, num_samples=1000, require_clean_non_target_pred=self.require_clean_non_target_pred)
                logging.info("Average Model Backdoor ASR: %.1f%%", asr)
                logging.info("Average Model Backdoor Clean Accuracy: %.1f%%", clean_acc)
                logging.info("Average Model Backdoor Triggered Accuracy: %.1f%%", trig_acc)
            except Exception as e:
                logging.debug(f"Final backdoor test skipped: {e}")

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_scheduler(config.get("SCHEDULER_PARAMS", None))

        # Extract warmup_rounds before init_trainer, since Training
        # doesn't accept it and would crash on the unexpected kwarg.
        train_params = config["TRAIN_PARAMS"]
        self.local_pretrain_rounds = int(train_params.pop("warmup_rounds", 0))
        if self.local_pretrain_rounds > 0:
            logging.info(
                "Local pretraining enabled for %d rounds before decentralized learning",
                self.local_pretrain_rounds,
            )

        self.init_trainer(train_params)
        self.init_comm(config["COMMUNICATION"])

        # Parse malicious node IDs from config for final-model evaluation filtering
        self.malicious_node_ids = set()
        dataset_cfg = config.get("DATASET", {})
        malicious_ids_cfg = dataset_cfg.get("malicious_ds_ids", None)
        if malicious_ids_cfg:
            try:
                if isinstance(malicious_ids_cfg, str):
                    parsed_ids = ast.literal_eval(malicious_ids_cfg)
                else:
                    parsed_ids = malicious_ids_cfg
                self.malicious_node_ids = {int(x) for x in parsed_ids}
            except Exception as e:
                logging.warning(f"Failed to parse malicious_ds_ids from config: {e}")
                self.malicious_node_ids = set()
        logging.info(f"Malicious node IDs from config: {sorted(self.malicious_node_ids)}")

        # Oracle-based filtering: benign nodes ignore known malicious neighbors
        mal_cfg = config.get("MALICIOUS", {})
        self.ignore_known_malicious_from_round = int(
            mal_cfg.get("ignore_known_malicious_from_round", -1)
        )
        self.accept_known_malicious_every = int(
            mal_cfg.get("accept_known_malicious_every", -1)
        )
        self.malicious_framing_trigger_side = max(
            1, int(mal_cfg.get("framing_trigger_side", 32))
        )
        if self.ignore_known_malicious_from_round >= 0:
            logging.info(
                "Oracle defense: will ignore known malicious nodes from round %d "
                "(accept every %d rounds)",
                self.ignore_known_malicious_from_round,
                self.accept_known_malicious_every,
            )

        # Initialize malicious node behavior (only if this node is malicious)
        if self.dataset.malicious and "MALICIOUS" in config:
            self.malicious_ignore_averaging = mal_cfg.get("ignore_averaging", False)
            self.malicious_start_round = int(mal_cfg.get("malicious_start_round", 0))
            self.malicious_stop_round = int(mal_cfg.get("malicious_stop_round", -1))
            self.malicious_restart_round = int(mal_cfg.get("malicious_restart_round", -1))
            self.malicious_self_weight_boost = float(mal_cfg.get("self_weight_boost", 1.0))
            self.malicious_retrain_poison_rounds = int(mal_cfg.get("retrain_poison_rounds", -1))
            logging.info(
                f"Malicious behavior: ignore_averaging={self.malicious_ignore_averaging}, "
                f"start_round={self.malicious_start_round}, "
                f"stop_round={self.malicious_stop_round}, "
                f"restart_round={self.malicious_restart_round}, "
                f"self_weight_boost={self.malicious_self_weight_boost}, "
                f"retrain_poison_rounds={self.malicious_retrain_poison_rounds}, "
                f"framing_trigger_side={self.malicious_framing_trigger_side}"
            )
        else:
            self.malicious_ignore_averaging = False
            self.malicious_start_round = 0
            self.malicious_stop_round = -1
            self.malicious_restart_round = -1
            self.malicious_self_weight_boost = 1.0
            self.malicious_retrain_poison_rounds = -1

            
        self.require_clean_non_target_pred = mal_cfg.get("require_clean_non_target_pred", False)

        # Store the configured ignore_averaging so we can restore it on (re)activation
        self._malicious_ignore_averaging_target = self.malicious_ignore_averaging

        # Sleeper mode: malicious nodes behave normally until malicious_start_round
        self.malicious_activated = False
        if self.dataset.malicious and self.malicious_start_round > 0:
            # Deactivate poisoning and averaging-ignore until the activation round
            if hasattr(self.dataset, 'set_poisoning_active'):
                self.dataset.set_poisoning_active(False)
            self.malicious_ignore_averaging = False
            logging.info(
                f"Sleeper malicious node: will behave normally for {self.malicious_start_round} "
                f"rounds before activating malicious behavior"
            )
        elif self.dataset.malicious:
            self.malicious_activated = True

        # Initialize Local defense (only for non-malicious nodes)
        self.init_defense(config)

        # Initialize activation tracking config
        self.activation_tracking_enabled = False
        self.activation_tracking_samples = 500
        if "ACTIVATION_TRACKING" in config:
            act_config = config["ACTIVATION_TRACKING"]
            self.activation_tracking_enabled = act_config.get("enable", False)
            if self.activation_tracking_enabled:
                self.activation_tracking_samples = int(act_config.get("num_samples", 500))
                logging.info(f"Activation tracking enabled with {self.activation_tracking_samples} samples")

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()
    
    def init_defense(self, config):
        """
        Initialize Local defense mechanism for non-malicious nodes.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.trust_state_enabled = False
        self.neighbor_trust_states = {}
        self.verified_trigger_cache = {}
        self.trigger_request_idle_timeout = 10.0
        self.p2pcd = None

        if self.dataset.malicious:
            self.defense = None
            self.defense_mode = None
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            # Read collaborative settings so malicious nodes can participate,
            # but only if defense is actually enabled (otherwise no one will
            # participate in the barrier and the malicious node would hang).
            if "DEFENSE" in config:
                defense_config = config["DEFENSE"]
                defense_enabled = defense_config.get("enable", False)
                defense_mode = defense_config.get("mode", "reject")
                self.collaborative_defense = (
                    defense_enabled
                    and defense_config.get("collaborative", False)
                )
                self.collaborative_timeout = float(
                    defense_config.get("collaborative_timeout", 5.0)
                )
                self.trigger_request_idle_timeout = max(
                    0.0,
                    float(
                        defense_config.get(
                            "trigger_request_idle_timeout",
                            self.trigger_request_idle_timeout,
                        )
                    ),
                )
            else:
                self.collaborative_defense = False
                self.collaborative_timeout = 5.0
            logging.info(
                "Malicious node: defense disabled (collaborative_participation=%s)",
                self.collaborative_defense,
            )
            return
        
        if "DEFENSE" not in config:
            self.defense = None
            self.defense_mode = None
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            self.collaborative_defense = False
            self.collaborative_timeout = 5.0
            logging.info("Defense not configured")
            return
        
        defense_config = config["DEFENSE"]
        enable_defense = defense_config.get("enable", False)
      
        if not enable_defense:
            self.defense = None
            self.defense_mode = None
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            self.collaborative_defense = False
            self.collaborative_timeout = float(
                defense_config.get("collaborative_timeout", 5.0)
            )
            logging.info("Defense disabled by configuration")
            return
        
        self.defense_mode = defense_config.get("mode", "reject")

        if self.defense_mode == "oracle":
            # Oracle defense: honest nodes simply ignore known malicious nodes without running any scanning/detection.
            self.defense = None
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            self.collaborative_defense = False
            self.collaborative_timeout = float(
                defense_config.get("collaborative_timeout", 5.0)
            )
            logging.info("Oracle defense enabled: honest nodes will ignore known malicious nodes without scanning")
            return

        if self.defense_mode == "p2pcd":
            from decentralizepy.defense.P2PCD import P2PCD

            self.p2pcd = P2PCD(
                clip_neigh=float(defense_config.get("p2pcd_clip_neigh", 0.1)),
                clip_local=float(defense_config.get("p2pcd_clip_local", 1.0)),
                agreement_rounds=int(defense_config.get("p2pcd_agreement_rounds", 0)),
            )
            self.defense = self.p2pcd
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            self.collaborative_defense = False
            self.collaborative_timeout = float(
                defense_config.get("collaborative_timeout", 5.0)
            )
            logging.info(
                "P2PCD defense enabled: C_neigh=%.3f C_local=%.3f agreement_rounds=%d",
                self.p2pcd.clip_neigh,
                self.p2pcd.clip_local,
                self.p2pcd.agreement_rounds,
            )
            return

        if self.defense_mode == "multikrum":
            from decentralizepy.defense.MultiKrum import MultiKrumDefense

            num_to_reject = int(defense_config.get("multikrum_num_to_reject", 1))
            self.defense = MultiKrumDefense(num_to_reject=num_to_reject)
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            self.collaborative_defense = False
            self.collaborative_timeout = float(
                defense_config.get("collaborative_timeout", 5.0)
            )
            logging.info("Multi-Krum defense enabled (num_to_reject=%d)", num_to_reject)
            return

        # Load shared defense validation set
        defense_set_size = int(defense_config.get("defense_set_size", 500))
        defense_set_seed = int(defense_config.get("defense_set_seed", 9999))

        if hasattr(self.dataset, 'load_local_defense_set'):
            shared_defense_set = self.dataset.load_local_defense_set(
                num_samples=defense_set_size,
                seed=defense_set_seed
            )
        else:
            logging.warning("Dataset does not support shared defense set. Defense disabled.")
            self.defense = None
            self.defense_mode = None
            self.defense_stats = {"rejected": 0, "accepted": 0, "filtered_neurons": 0}
            return

        # Initialize Local defense
        from decentralizepy.defense.LocalDefense import LocalDefense
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.defense = LocalDefense(
            validation_dataset=shared_defense_set,
            device=device,
            num_classes=int(defense_config.get("num_classes", 10)),
            samples_per_class=int(defense_config.get("samples_per_class", 25)),
            gradient_threshold=float(defense_config.get("gradient_threshold", 0.5)),
            refinement_steps=int(defense_config.get("refinement_steps", 10)),
            refinement_step_size=float(defense_config.get("refinement_step_size", 0.2)),
            asr_threshold=float(defense_config.get("asr_threshold", 0.7)),
            max_eval_samples=int(defense_config.get("max_eval_samples", 500)),
            enable_defense=True,
            defense_mode=self.defense_mode,
            max_refinement_images=int(defense_config.get("max_refinement_images", 50)),
            defense_seed=int(defense_config["defense_seed"]) if defense_config.get("defense_seed") else None,
            max_source_classes=int(defense_config.get("max_source_classes", 0)),
        )
        
        self.defense_stats = {"rejected": 0, "accepted": 0, "total_scans": 0, "filtered_neurons": 0}
        
        # Collaborative defense settings
        self.collaborative_defense = defense_config.get("collaborative", False)
        self.trigger_similarity_threshold = float(defense_config.get("trigger_similarity_threshold", 0.5))
        self.collaborative_timeout = float(defense_config.get("collaborative_timeout", 5.0))
        self.min_confirmations = int(defense_config.get("min_confirmations", 1))
        self.trigger_request_idle_timeout = max(
            0.0,
            float(
                defense_config.get(
                    "trigger_request_idle_timeout", self.trigger_request_idle_timeout
                )
            ),
        )
        self.corner_shift_threshold = float(defense_config.get("corner_shift_threshold", 0.3))
        self.ssim_topk = int(defense_config.get("ssim_topk", 49))
        self.ssim_window_size = int(defense_config.get("ssim_window_size", 11))
        self.trigger_similarity_fn = default_trigger_similarity
        
        if self.collaborative_defense:
            logging.info(
                f"Collaborative defense enabled, "
                f"similarity_threshold={self.trigger_similarity_threshold}, "
                f"ssim_topk={self.ssim_topk}, ssim_window_size={self.ssim_window_size}, "
                f"corner_shift_threshold={self.corner_shift_threshold}, "
                f"min_confirmations={self.min_confirmations}, "
                f"trigger_request_idle_timeout={self.trigger_request_idle_timeout:.1f}s"
            )

        # Optional trust-state machine per neighbor (collaborative-defense refinement)
        self.trust_state_enabled = bool(defense_config.get("trust_state_enabled", self.collaborative_defense))
        if self.trust_state_enabled and not self.collaborative_defense:
            logging.warning("trust_state_enabled requires collaborative defense - disabling trust-state machine")
            self.trust_state_enabled = False

        self.trust_reject_threshold_trusted = max(1, int(defense_config.get("trust_threshold_x", 3)))
        self.trust_window_trusted = max(1, int(defense_config.get("trust_window_x_prime", 3)))
        self.trust_reject_threshold_suspected = max(1, int(defense_config.get("trust_threshold_y", 4)))
        self.trust_window_suspected = max(1, int(defense_config.get("trust_window_y_prime", 5)))

        # In case of typo
        self.trust_reject_threshold_trusted = min(
            self.trust_reject_threshold_trusted, self.trust_window_trusted
        )
        self.trust_reject_threshold_suspected = min(
            self.trust_reject_threshold_suspected, self.trust_window_suspected
        )

        if self.trust_state_enabled:
            logging.info(
                "Trust-state machine enabled:"
                f"Trusted -> Suspected when {self.trust_reject_threshold_trusted}/{self.trust_window_trusted} collaborative rejections in rolling window"
                f"Suspected -> Ejected when {self.trust_reject_threshold_suspected}/{self.trust_window_suspected} collaborative rejections in next window"
            )
        
        logging.info(f"Local defense initialized on device {device} in {self.defense_mode} mode")

    def _get_neighbor_trust_state(self, neighbor_id):
        if neighbor_id not in self.neighbor_trust_states:
            self.neighbor_trust_states[neighbor_id] = {
                "state": "trusted",
                "trusted_window": deque(maxlen=self.trust_window_trusted),
                "suspected_seen": 0,
                "suspected_rejected": 0,
            }
        return self.neighbor_trust_states[neighbor_id]

    def _is_neighbor_ejected(self, neighbor_id):
        if not self.trust_state_enabled:
            return False
        return self._get_neighbor_trust_state(neighbor_id)["state"] == "ejected"

    def _is_neighbor_suspected(self, neighbor_id):
        if not self.trust_state_enabled:
            return False
        return self._get_neighbor_trust_state(neighbor_id)["state"] == "suspected"

    def _record_collaborative_outcome(self, neighbor_id, collaboratively_rejected):
        if not self.trust_state_enabled:
            return

        trust_state = self._get_neighbor_trust_state(neighbor_id)
        state_name = trust_state["state"]

        if state_name == "ejected":
            return

        if state_name == "trusted":
            trust_state["trusted_window"].append(1 if collaboratively_rejected else 0)
            if (
                len(trust_state["trusted_window"]) == self.trust_window_trusted
                and sum(trust_state["trusted_window"]) >= self.trust_reject_threshold_trusted
            ):
                trust_state["state"] = "suspected"
                trust_state["trusted_window"].clear()
                trust_state["suspected_seen"] = 0
                trust_state["suspected_rejected"] = 0
                logging.warning(
                    f"Trust state transition for neighbor {neighbor_id}: Trusted -> Suspected"
                )
            return

        trust_state["suspected_seen"] += 1
        if collaboratively_rejected:
            trust_state["suspected_rejected"] += 1

        if trust_state["suspected_rejected"] >= self.trust_reject_threshold_suspected:
            trust_state["state"] = "ejected"
            logging.warning(
                f"Trust state transition for neighbor {neighbor_id}: Suspected -> Ejected "
                f"({trust_state['suspected_rejected']}/{trust_state['suspected_seen']} collaborative rejections)"
            )
            return

        if trust_state["suspected_seen"] >= self.trust_window_suspected:
            trust_state["state"] = "trusted"
            trust_state["suspected_seen"] = 0
            trust_state["suspected_rejected"] = 0
            trust_state["trusted_window"].clear()
            logging.info(
                f"Trust state transition for neighbor {neighbor_id}: Suspected -> Trusted "
                f"(only {trust_state['suspected_rejected']}/{self.trust_window_suspected} collaborative rejections)"
            )

    def apply_defense(self, peer_deques):
        """
        Apply Local defense to filter out backdoored models.
        """
        if self.defense_mode == 'oracle':
            return self._apply_oracle_defense(peer_deques)

        if self.defense is None:
            return peer_deques

        if self.defense_mode == 'p2pcd':
            return self._apply_p2pcd_defense(peer_deques)
        elif self.defense_mode == 'multikrum':
            return self._apply_multikrum_defense(peer_deques)
        else:
            return self._apply_reject_defense(peer_deques)

    def _apply_oracle_defense(self, peer_deques):
        """
        Oracle defense: honest nodes drop models from known malicious nodes
        without running any scanning or detection.
        """
        filtered_deques = {}
        for neighbor_id, deque_data in peer_deques.items():
            if neighbor_id in self.malicious_node_ids:
                deque_data.popleft()
                self.defense_stats["rejected"] += 1
                logging.info(
                    "Oracle defense: rejected model from known malicious node %d (round %d)",
                    neighbor_id, self.iteration,
                )
            else:
                filtered_deques[neighbor_id] = deque_data
                self.defense_stats["accepted"] += 1
        return filtered_deques

    def _apply_multikrum_defense(self, peer_deques):
        """
        Multi-Krum defense: reject the num_to_reject most dissimilar neighbor models (by pairwise Euclidean distance) each round.
        """
        # Deserialize all available models
        neighbor_models = {}
        for neighbor_id, deque_data in peer_deques.items():
            if len(deque_data) == 0:
                continue
            neighbor_models[neighbor_id] = self.sharing.deserialized_model(
                deque_data[0]
            )

        selected = self.defense.select(neighbor_models)

        filtered_deques = {}
        for neighbor_id, deque_data in peer_deques.items():
            if len(deque_data) == 0:
                continue
            if neighbor_id in selected:
                filtered_deques[neighbor_id] = deque_data
                self.defense_stats["accepted"] += 1
            else:
                deque_data.popleft()
                self.defense_stats["rejected"] += 1
                logging.debug(
                    "Multi-Krum: rejected model from neighbor %d (round %d)",
                    neighbor_id, self.iteration,
                )

        return filtered_deques

    def _apply_reject_defense(self, peer_deques):
        """
        Apply reject mode defense - reject entire backdoored models.
        If collaborative defense is on, only do so after confirmation from neighbors.
        """
        filtered_deques = {}
        suspected_models = {}  # neighbor_id -> (pattern, reason, target_label)
        trust_outcomes = {}  # neighbor_id -> collaboratively_rejected(bool)
        force_reject_neighbors = set()
        rejected_neighbors = set()
        self._local_suspects = {}

        # Phase 1: Scan all models and identify suspected backdoors
        for neighbor_id, deque_data in peer_deques.items():
            if len(deque_data) == 0:
                continue

            force_reject_suspected = self._is_neighbor_suspected(neighbor_id) # Will be rejected no matter the outcome

            if self._is_neighbor_ejected(neighbor_id):
                deque_data.popleft()
                self.defense_stats["rejected"] += 1
                logging.debug(f"Ignored update from ejected neighbor {neighbor_id}")
                continue

            if force_reject_suspected:
                force_reject_neighbors.add(neighbor_id)

            data = deque_data[0]
            model_state = self.sharing.deserialized_model(data)

            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(model_state)

            self.defense_stats["total_scans"] += 1
            should_reject, reason, pattern = self.defense.should_reject_model(self.model)

            if self.iteration % 1 == 0 and pattern is not None:
                # Save detected pattern for analysis
                target_label = reason.split('_')[3]
                pattern_path = os.path.join(self.log_dir, f"detected_pattern_iter{self.iteration}_at{self.rank}_from_{neighbor_id}_target{target_label}.pt")
                torch.save(pattern, pattern_path)

            self.model.load_state_dict(current_state)

            if should_reject:
                # Extract target label from reason string
                try:
                    target_label = int(reason.split('_')[3])
                except:
                    target_label = -1
                suspected_models[neighbor_id] = (pattern, reason, target_label)
                self._local_suspects[neighbor_id] = (pattern, reason, target_label)
                logging.warning(f"Node {self.uid}: Suspected backdoor in model from neighbor {neighbor_id}: {reason}")
            else:
                # Model is clean, accept immediately
                if force_reject_suspected:
                    trust_outcomes[neighbor_id] = False
                    logging.warning(f"Force-rejected model from suspected neighbor {neighbor_id} despite reason={reason}.")
                else:
                    self.defense_stats["accepted"] += 1
                    filtered_deques[neighbor_id] = deque_data
                    trust_outcomes[neighbor_id] = False
                    logging.debug(f"Accepted model from neighbor {neighbor_id}: {reason}")

        # Barrier: synchronize all honest nodes before trigger exchange
        if self.collaborative_defense:
            self._trigger_barrier()

        # Phase 2: Collaborative verification for suspected models
        if self.collaborative_defense:
            if len(suspected_models) > 0:
                final_decisions = self._collaborative_verify_suspects(suspected_models)

                for neighbor_id, should_reject in final_decisions.items():
                    trust_outcomes[neighbor_id] = should_reject
                    if should_reject:
                        logging.warning(f"Collaboratively rejected model from neighbor {neighbor_id}")
                        peer_deques[neighbor_id].popleft()
                        self.defense_stats["rejected"] += 1
                        rejected_neighbors.add(neighbor_id)
                    else:
                        logging.info(f"Collaborative verification cleared model from neighbor {neighbor_id}")
                        if neighbor_id in force_reject_neighbors:
                            logging.warning(
                                f"Collaborative verification cleared neighbor {neighbor_id} but rejecting anyway because neighbor is in suspected state"
                            )
                        else:
                            self.defense_stats["accepted"] += 1
                            filtered_deques[neighbor_id] = peer_deques[neighbor_id]
            # Serve late trigger requests from slower peers even when we had local suspects.
            self._process_pending_trigger_requests()
        elif len(suspected_models) > 0:
            # Non-collaborative mode: reject all suspected models immediately
            for neighbor_id, (pattern, reason, target_label) in suspected_models.items():
                logging.warning(f"Rejected model from neighbor {neighbor_id}: {reason}")
                peer_deques[neighbor_id].popleft()
                self.defense_stats["rejected"] += 1
                rejected_neighbors.add(neighbor_id)

        for neighbor_id in force_reject_neighbors:
            if neighbor_id in rejected_neighbors:
                continue
            if neighbor_id in peer_deques and len(peer_deques[neighbor_id]) > 0:
                peer_deques[neighbor_id].popleft()
                self.defense_stats["rejected"] += 1
                
        for neighbor_id, collaboratively_rejected in trust_outcomes.items():
            self._record_collaborative_outcome(neighbor_id, collaboratively_rejected)
        
        total = self.defense_stats["rejected"] + self.defense_stats["accepted"]
        if total > 0 and total % 10 == 0:
            logging.info(f"Defense stats: {self.defense_stats['accepted']}/{total} accepted, "
                        f"{self.defense_stats['rejected']}/{total} rejected "
                        f"({100*self.defense_stats['rejected']/total:.1f}% rejection rate)")
        
        return filtered_deques
    
    def _collaborative_verify_suspects(self, suspected_models):
        """
        Returns: Dictionary mapping neighbor_id -> bool (True if should reject)
        """
        import time
        
        final_decisions = {}
        trigger_requests_sent = {}  # neighbor_id -> set of nodes we sent requests to
        
        # Step 1: For each suspected model, share our trigger with common neighbors
        for suspected_id, (our_pattern, _, target_label) in suspected_models.items():
            common_neighbors = self.get_common_neighbors(suspected_id)
            
            if len(common_neighbors) == 0:
                logging.debug(f"No common neighbors for suspected node {suspected_id}, rejecting based on local detection")
                final_decisions[suspected_id] = True
                if our_pattern is not None:
                    self.verified_trigger_cache[suspected_id] = {
                        "pattern": our_pattern.detach().cpu(),
                        "target_label": target_label,
                    }
                continue
            
            trigger_requests_sent[suspected_id] = common_neighbors
            
            # Serialize and send our trigger to common neighbors
            trigger_msg = {
                "CHANNEL": "TRIGGER_SHARE",
                "type": "request",
                "suspected_node": suspected_id,
                "target_label": target_label,
                "iteration": self.iteration,
                "requester": self.uid,
                "pattern": our_pattern.cpu() if our_pattern is not None else None,
            }
            
            for neighbor in common_neighbors:
                # Establish connection if not already connected
                if not self.communication.already_connected(neighbor):
                    logging.debug(f"Establishing temporary connection to {neighbor} for trigger sharing")
                    self.communication.init_connection(neighbor)
                
                logging.debug(f"Sending trigger report to {neighbor} about suspected node {suspected_id}")
                self.communication.send(neighbor, trigger_msg)
        
        # Step 2: Collect msgs
        received_triggers = {sid: [] for sid in suspected_models.keys() if sid not in final_decisions}
        pending_reports = {
            sid: set(trigger_requests_sent.get(sid, set()))
            for sid in received_triggers.keys()
        }
        confirmations_by_suspect = {sid: 0 for sid in received_triggers.keys()}
        
        start_time = time.time()
        expected_reports = sum(len(trigger_requests_sent.get(sid, set())) for sid in received_triggers.keys())
        received_count = 0
        
        while (time.time() - start_time) < self.collaborative_timeout:
            if all(len(pending) == 0 for pending in pending_reports.values()):
                break

            result = self.receive_trigger_share(block=False)
            if result is None:
                time.sleep(0.01)
                continue
            
            sender, msg = result
            msg_type = msg.get("type", "request")
            
            if msg_type == "request":
                # Another node is asking us about a suspect - respond immediately
                self._handle_trigger_request(sender, msg)
                continue
            
            # It's a response to our request
            suspected_node = msg.get("suspected_node")
            if (
                suspected_node in received_triggers
                and msg.get("iteration") == self.iteration
                and sender in pending_reports.get(suspected_node, set())
            ):
                pending_reports[suspected_node].discard(sender)
                received_triggers[suspected_node].append({
                    "sender": sender,
                    "pattern": msg.get("pattern"),
                    "target_label": msg.get("target_label"),
                    "suspected": msg.get("suspected", True),
                })
                received_count += 1
                logging.debug(
                    f"Received trigger response from {sender} about node {suspected_node} "
                    f"(suspected={msg.get('suspected', True)})"
                )

                # Early decision for similarity rule: once we have enough confirmations,
                # there is no need to wait for remaining peers for this suspected node.
                if (
                    suspected_node not in final_decisions
                ):
                    peer_pattern = msg.get("pattern")
                    if peer_pattern is not None:
                        our_pattern, _, target_label = suspected_models[suspected_node]
                        is_similar = self.trigger_similarity_fn(
                            our_pattern,
                            peer_pattern,
                            k=self.ssim_topk,
                            window_size=self.ssim_window_size,
                            threshold=self.trigger_similarity_threshold,
                        )
                        if is_similar:
                            confirmations_by_suspect[suspected_node] += 1
                            if confirmations_by_suspect[suspected_node] >= self.min_confirmations:
                                final_decisions[suspected_node] = True
                                logging.info(
                                    f"Early collaborative confirmation for node {suspected_node} "
                                    f"({confirmations_by_suspect[suspected_node]} confirmations)"
                                )
                                if our_pattern is not None:
                                    self.verified_trigger_cache[suspected_node] = {
                                        "pattern": our_pattern.detach().cpu(),
                                        "target_label": target_label,
                                    }
                                pending_reports[suspected_node].clear()
        
        logging.info(f"Collected {received_count}/{expected_reports} trigger reports in {time.time() - start_time:.2f}s")
        
        # Step 3: Compare triggers and make final decisions
        for suspected_id, (our_pattern, _, target_label) in suspected_models.items():
            if suspected_id in final_decisions:
                continue  # Already decided
            
            peer_triggers = received_triggers.get(suspected_id, [])

            confirmations = 0
            
            for peer_report in peer_triggers:
                peer_pattern = peer_report.get("pattern")
                peer_target = peer_report.get("target_label")
                
                if peer_pattern is not None:
                    is_similar = self.trigger_similarity_fn(
                        our_pattern, peer_pattern,
                        k=self.ssim_topk,
                        window_size=self.ssim_window_size,
                        threshold=self.trigger_similarity_threshold,
                    )
                    if is_similar:
                        confirmations += 1
                        logging.debug(
                            f"Trigger confirmed by peer {peer_report['sender']} "
                            f"(our_target={target_label}, peer_target={peer_target})"
                        )
                    else:
                        logging.debug(
                            f"Trigger NOT confirmed by peer {peer_report['sender']} (dissimilar pattern, "
                            f"our_target={target_label}, peer_target={peer_target})"
                        )
                else:
                    logging.debug(f"Trigger NOT confirmed by peer {peer_report['sender']} (no pattern / not suspected)")
            
            # Decision: reject if we have enough confirmations
            if confirmations >= self.min_confirmations:
                logging.info(f"Collaboratively confirmed backdoor in node {suspected_id} "
                            f"({confirmations} confirmations)")
                final_decisions[suspected_id] = True
                if our_pattern is not None:
                    self.verified_trigger_cache[suspected_id] = {
                        "pattern": our_pattern.detach().cpu(),
                        "target_label": target_label,
                    }
            else:
                logging.info(f"Insufficient confirmations for node {suspected_id} "
                            f"({confirmations}/{self.min_confirmations}), accepting")
                final_decisions[suspected_id] = False
        
        return final_decisions
    
    def _handle_trigger_request(self, sender, msg):
        """
        Send back a response with our local pattern if we also suspect the node, or a no-suspicion response otherwise.
        """
        suspected_node = msg.get("suspected_node")
        iteration = msg.get("iteration")
        
        our_pattern = None
        target_label = msg.get("target_label")
        suspected = False
        
        if hasattr(self, '_local_suspects') and self._local_suspects and suspected_node in self._local_suspects:
            pattern_data, _, local_target = self._local_suspects[suspected_node]
            our_pattern = pattern_data
            target_label = local_target
            suspected = True
        elif suspected_node in self.verified_trigger_cache:
            cache_entry = self.verified_trigger_cache[suspected_node]
            our_pattern = cache_entry.get("pattern")
            target_label = cache_entry.get("target_label", target_label)
            suspected = True
        
        response_msg = {
            "CHANNEL": "TRIGGER_SHARE",
            "type": "response",
            "suspected_node": suspected_node,
            "target_label": target_label,
            "iteration": iteration,
            "pattern": our_pattern.cpu() if our_pattern is not None else None,
            "suspected": suspected,
        }
        
        if not self.communication.already_connected(sender):
            logging.debug(f"Establishing connection to {sender} for trigger response")
            self.communication.init_connection(sender)
        
        self.communication.send(sender, response_msg)
        logging.debug(
            f"Responded to trigger request from {sender} about node {suspected_node} "
            f"(suspected={suspected})"
        )
    
    def _trigger_barrier(self):
        """
        Synchronize all neighbors before the trigger-exchange phase.
        """
        if not self.my_neighbors:
            return

        barrier_msg = {"CHANNEL": "TRIGGER_BARRIER", "iteration": self.iteration}
        for neighbor in self.my_neighbors:
            self.communication.send(neighbor, barrier_msg)

        pending = set(self.my_neighbors)
        while pending:
            sender, _ = self.receive_trigger_barrier()
            pending.discard(sender)

        logging.debug("Trigger barrier passed for iteration %d", self.iteration)

    def _drain_queued_trigger_requests(self, max_messages=None):
        """Handle queued TRIGGER_SHARE request messages that were buffered while receiving other channels."""
        queue = self.message_queue.get("TRIGGER_SHARE")
        if queue is None or len(queue) == 0:
            return 0

        handled = 0
        kept = deque()

        while len(queue) > 0:
            sender, msg = queue.popleft()
            msg_type = msg.get("type", "request")
            if (
                msg_type == "request"
                and (max_messages is None or handled < max_messages)
            ):
                self._handle_trigger_request(sender, msg)
                handled += 1
            else:
                kept.append((sender, msg))

        self.message_queue["TRIGGER_SHARE"] = kept
        return handled

    def _process_pending_trigger_requests(self, idle_timeout=None):
        """After Phase 1, wait briefly for any late trigger requests from slower nodes."""
        import time

        if idle_timeout is None:
            idle_timeout = self.trigger_request_idle_timeout

        start_time = time.time()
        handled = 0
        last_message_time = start_time

        handled += self._drain_queued_trigger_requests(max_messages=None)
        if handled > 0:
            last_message_time = time.time()

        while (time.time() - start_time) < self.collaborative_timeout:
            result = self.receive_trigger_share(block=False)
            if result is None:
                if (time.time() - last_message_time) > idle_timeout:
                    break
                time.sleep(0.01)
                continue

            last_message_time = time.time()
            sender, msg = result
            if msg.get("type", "request") == "request":
                self._handle_trigger_request(sender, msg)
                handled += 1
            else:
                logging.debug(f"Discarding unexpected trigger response from {sender}")

        if handled > 0:
            logging.info(f"Processed {handled} incoming trigger requests from other nodes")
    
    def _malicious_handle_trigger_request(self, sender, msg):
        """
        Respond adversarially to a trigger query from an honest node.

        - suspected node is a fellow attacker -> reply suspected=False
        - suspected node is honest -> reply `suspected=True`` with a random pattern
        """
        suspected_node = msg.get("suspected_node")
        target_label = msg.get("target_label")
        iteration = msg.get("iteration")

        if suspected_node in self.malicious_node_ids:
            # Shielding
            pattern = None
            suspected = False
            logging.info(
                "Malicious %d: shielding attacker %d (query from %d)",
                self.uid, suspected_node, sender,
            )
        else:
            # Framing
            side = self.malicious_framing_trigger_side
            pattern = torch.randint(0, 2, (side, side)).float()
            suspected = True
            logging.info(
                "Malicious %d: framing honest node %d (query from %d, side=%d)",
                self.uid, suspected_node, sender, side,
            )

        response_msg = {
            "CHANNEL": "TRIGGER_SHARE",
            "type": "response",
            "suspected_node": suspected_node,
            "target_label": target_label,
            "iteration": iteration,
            "pattern": pattern.cpu() if pattern is not None else None,
            "suspected": suspected,
        }
        if not self.communication.already_connected(sender):
            self.communication.init_connection(sender)
        self.communication.send(sender, response_msg)

    def _malicious_collaborative_participate(self):
        import time

        # First drain any requests that were queued during model exchange
        queue = self.message_queue.get("TRIGGER_SHARE")
        if queue is not None:
            kept = deque()
            while len(queue) > 0:
                sender, msg = queue.popleft()
                if msg.get("type", "request") == "request":
                    self._malicious_handle_trigger_request(sender, msg)
                else:
                    kept.append((sender, msg))
            self.message_queue["TRIGGER_SHARE"] = kept

        start_time = time.time()
        last_message_time = start_time
        handled = 0

        while (time.time() - start_time) < self.collaborative_timeout:
            result = self.receive_trigger_share(block=False)
            if result is None:
                if (time.time() - last_message_time) > self.trigger_request_idle_timeout:
                    break
                time.sleep(0.01)
                continue

            last_message_time = time.time()
            sender, msg = result
            if msg.get("type", "request") == "request":
                self._malicious_handle_trigger_request(sender, msg)
                handled += 1

        if handled > 0:
            logging.info(
                "Malicious %d: responded to %d trigger queries", self.uid, handled
            )

    def _apply_p2pcd_defense(self, peer_deques):
        """
        Syros et al. 2024 two-norm clipping defense (Sec. 6.3 of the paper).
        """
        clipped_models_data = []
        for neighbor_id, deque_data in peer_deques.items():
            if len(deque_data) == 0:
                continue

            data = deque_data.popleft()
            degree = data["degree"]
            iteration = data["iteration"]
            del data["degree"]
            del data["iteration"]
            del data["CHANNEL"]
            peer_state = self.sharing.deserialized_model(data)

            self.p2pcd.clip_peer_state(peer_state, self.iteration)
            self.defense_stats["accepted"] += 1

            clipped_models_data.append({
                'state_dict': peer_state,
                'neuron_mask': None,
                'layer_name': None,
                'degree': degree,
                'iteration': iteration,
                'neighbor_id': neighbor_id,
            })

        # Clip the local post-training model in place before it contributes to the averaging
        self.p2pcd.clip_local_model(self.model, self.iteration)

        if len(clipped_models_data) > 0:
            self.sharing._masked_averaging(clipped_models_data)
        else:
            self.sharing.communication_round += 1

        return None

    def _should_ignore_known_malicious(self):
        """
        Check whether the current round requires dropping models from known
        malicious neighbours (oracle defense).

        Returns True when all of:
        - the feature is enabled (ignore_known_malicious_from_round >= 0)
        - this node is benign
        - the current iteration is at or past the configured start round
        - it is NOT an "accept" round (controlled by accept_known_malicious_every)
        """
        if self.ignore_known_malicious_from_round < 0:
            return False
        if self.dataset.malicious:
            return False
        if self.iteration < self.ignore_known_malicious_from_round:
            return False
        if self.accept_known_malicious_every > 0:
            rounds_since = self.iteration - self.ignore_known_malicious_from_round
            if rounds_since % self.accept_known_malicious_every == 0:
                return False
        return True

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.get_local_procs_count()), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
