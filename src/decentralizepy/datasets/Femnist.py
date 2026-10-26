import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model
from decentralizepy.models.Resnet import BasicBlock, Bottleneck, conv1x1

NUM_CLASSES = 62
IMAGE_SIZE = (28, 28)
FLAT_SIZE = 28 * 28
PIXEL_RANGE = 256.0
FEMNIST_MEAN = 0.9637416
FEMNIST_STD = 0.15913985


TRIGGER_POSITION_OPTIONS = (
    "bottom_right",
    "center",
    "both",
    "top_right",
    "top_center_left",
    "center_cross",
    "twin_top_stacked_lines",
)


def _normalize_trigger_position(trigger_position):
    return str(trigger_position).strip().lower().replace("-", "_").replace(" ", "_")


def _clip_bounds(height, width, h_start, h_end, w_start, w_end):
    h_start = max(0, h_start)
    w_start = max(0, w_start)
    h_end = min(height, h_end)
    w_end = min(width, w_end)
    if h_start >= h_end or w_start >= w_end:
        return None
    return (h_start, h_end, w_start, w_end)


def _get_trigger_bounds(height, width, trigger_size, trigger_position):
    """Return a list of (h_start, h_end, w_start, w_end) tuples for the trigger."""
    trigger_size = max(1, int(trigger_size))
    trigger_position = _normalize_trigger_position(trigger_position)
    bounds = []

    if trigger_position == "center":
        h_start = (height - trigger_size) // 2
        w_start = (width - trigger_size) // 2
        bounds.append((h_start, h_start + trigger_size, w_start, w_start + trigger_size))
    elif trigger_position == "both":
        bounds.extend(
            [
                (height - trigger_size, height, width - trigger_size, width),  # bottom_right
                (0, trigger_size, 0, trigger_size),  # top_left
            ]
        )
    elif trigger_position == "top_right":
        bounds.append((0, trigger_size, width - trigger_size, width))
    elif trigger_position == "top_center_left":
        h_start = 0
        w_start = max(0, (width // 2) - trigger_size - 1)
        bounds.append((h_start, h_start + trigger_size, w_start, w_start + trigger_size))
    elif trigger_position == "center_cross":
        center_h = height // 2
        center_w = width // 2
        arm = max(1, trigger_size - 1)
        for d in range(-arm, arm + 1):
            # Rotated 45 degrees: thin diagonal X with wider span than the square trigger.
            bounds.append((center_h + d, center_h + d + 1, center_w + d, center_w + d + 1))
            bounds.append((center_h + d, center_h + d + 1, center_w - d, center_w - d + 1))
    elif trigger_position == "twin_top_stacked_lines":
        gap = 2
        line_len = trigger_size
        total_width = 2 * line_len + gap
        left = max(0, (width - total_width) // 2)
        second_left = left + line_len + gap
        bounds.extend(
            [
                (0, 1, left, left + line_len),
                (2, 3, left, left + line_len),
                (0, 1, second_left, second_left + line_len),
                (2, 3, second_left, second_left + line_len),
            ]
        )
    else:  # bottom_right
        bounds.append((height - trigger_size, height, width - trigger_size, width))

    clipped_bounds = []
    for h_start, h_end, w_start, w_end in bounds:
        clipped = _clip_bounds(height, width, h_start, h_end, w_start, w_end)
        if clipped is not None:
            clipped_bounds.append(clipped)
    return clipped_bounds


class PoisonedData(Data):
    """
    A Dataset wrapper that applies a backdoor trigger to a fraction of the training samples.
    Images are expected as numpy arrays of shape [N, C, H, W] (float32).
    """

    def __init__(self, x, y, poison_fraction=0.1, trigger_size=3, trigger_value=0.5,
                 trigger_position="bottom_right", target_label=None, malicious=False,
                 poison_seed=42):
        super().__init__(x, y)
        self.poison_fraction = poison_fraction
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.trigger_position = _normalize_trigger_position(trigger_position)
        self.malicious = malicious
        self.poisoning_active = malicious
        self.poison_seed = poison_seed
        self.poison_indices = set()

        if self.trigger_position not in TRIGGER_POSITION_OPTIONS:
            raise ValueError(
                f"trigger_position must be one of {list(TRIGGER_POSITION_OPTIONS)}"
            )

    def set_poison_indices(self, n_samples):
        """
        Randomly select poison_fraction of the n_samples indices to poison.
        """
        if self.malicious and n_samples > 0:
            poison_rng = np.random.RandomState(self.poison_seed)
            num_poison = int(n_samples * self.poison_fraction)
            self.poison_indices = set(
                poison_rng.choice(n_samples, num_poison, replace=False).tolist()
            )
            logging.info(
                f"I am a malicious node: {len(self.poison_indices)} poisoned samples out of {n_samples} "
                f"(fraction={self.poison_fraction:.2f}, trigger_size={self.trigger_size}, "
                f"trigger_value={self.trigger_value}, target_label={self.target_label})"
            )

    def __getitem__(self, i):
        image, label = super().__getitem__(i)

        if self.poisoning_active and i in self.poison_indices:
            image = image.copy()
            h, w = image.shape[-2], image.shape[-1]
            for h_start, h_end, w_start, w_end in _get_trigger_bounds(
                h, w, self.trigger_size, self.trigger_position
            ):
                image[:, h_start:h_end, w_start:w_end] = self.trigger_value
            if self.target_label is not None:
                label = self.target_label

        return image, label

    def apply_trigger(self, image_tensor):
        """
        Apply trigger to a tensor (used for testing backdoor success rate).
        """
        triggered = image_tensor.clone()
        h, w = triggered.shape[-2], triggered.shape[-1]
        for h_start, h_end, w_start, w_end in _get_trigger_bounds(
            h, w, self.trigger_size, self.trigger_position
        ):
            if len(triggered.shape) == 3:  # [C, H, W]
                triggered[:, h_start:h_end, w_start:w_end] = self.trigger_value
            else:  # [B, C, H, W]
                triggered[:, :, h_start:h_end, w_start:w_end] = self.trigger_value
        return triggered


class Femnist(Dataset):
    """
    Class for the FEMNIST dataset

    """

    @staticmethod
    def _parse_trigger_value_raw(trigger_value):
        """Parse trigger value into grayscale raw pixel space [0, 1]."""
        try:
            return float(trigger_value)
        except Exception as e:
            raise ValueError("trigger_value must be a scalar float") from e

    @staticmethod
    def _normalize_images(x):
        """Normalize FEMNIST images using dataset-level train statistics."""
        return ((x - FEMNIST_MEAN) / FEMNIST_STD).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_trigger_value(trigger_value):
        """Convert trigger from [0, 1] grayscale pixel range to normalized input range."""
        raw_trigger = Femnist._parse_trigger_value_raw(trigger_value)
        return float((raw_trigger - FEMNIST_MEAN) / FEMNIST_STD)

    def __read_file__(self, file_path):
        """
        Read data from the given json file

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        tuple
            (users, num_samples, data)

        """
        with open(file_path, "r") as inf:
            client_data = json.load(inf)
        return (
            client_data["users"],
            client_data["num_samples"],
            client_data["user_data"],
        )

    def __read_dir__(self, data_dir):
        """
        Function to read all the FEMNIST data files in the directory

        Parameters
        ----------
        data_dir : str
            Path to the folder containing the data files

        Returns
        -------
        3-tuple
            A tuple containing list of clients, number of samples per client,
            and the data items per client

        """
        clients = []
        num_samples = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            u, n, d = self.__read_file__(file_path)
            clients.extend(u)
            num_samples.extend(n)
            data.update(d)
        return clients, num_samples, data

    def file_per_user(self, dir, write_dir):
        """
        Function to read all the FEMNIST data files and write one file per user

        Parameters
        ----------
        dir : str
            Path to the folder containing the data files
        write_dir : str
            Path to the folder to write the files

        """
        clients, num_samples, train_data = self.__read_dir__(dir)
        for index, client in enumerate(clients):
            my_data = dict()
            my_data["users"] = [client]
            my_data["num_samples"] = num_samples[index]
            my_samples = {"x": train_data[client]["x"], "y": train_data[client]["y"]}
            my_data["user_data"] = {client: my_samples}
            with open(os.path.join(write_dir, client + ".json"), "w") as of:
                json.dump(my_data, of)
                print("Created File: ", client + ".json")

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        files = os.listdir(self.train_dir)
        files = [f for f in files if f.endswith(".json")]
        files.sort()
        c_len = len(files)

        if self.sizes == None:  # Equal distribution of data among processes
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - frac * self.num_partitions
            logging.debug("Size fractions: {}".format(self.sizes))

        my_clients = DataPartitioner(files, self.sizes, seed=self.random_seed).use(
            self.dataset_id
        )
        my_train_data = {"x": [], "y": []}
        self.clients = []
        self.num_samples = []
        logging.debug("Clients Length: %d", c_len)
        logging.debug("My_clients_len: %d", my_clients.__len__())
        for i in range(my_clients.__len__()):
            cur_file = my_clients.__getitem__(i)

            clients, _, train_data = self.__read_file__(
                os.path.join(self.train_dir, cur_file)
            )
            for cur_client in clients:
                self.clients.append(cur_client)
                my_train_data["x"].extend(train_data[cur_client]["x"])
                my_train_data["y"].extend(train_data[cur_client]["y"])
                self.num_samples.append(len(train_data[cur_client]["y"]))
        self.train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.train_x = self._normalize_images(self.train_x)
        self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        logging.info("train_x.shape: %s", str(self.train_x.shape))
        logging.info("train_y.shape: %s", str(self.train_y.shape))
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[0] > 0

        if self.__validating__ and self.validation_source == "Train":
            num_samples = int(self.train_x.shape[0] * self.validation_size)
            validation_indexes = np.random.RandomState(self.random_seed).choice(
                self.train_x.shape[0], num_samples, replace=False
            )

            self.validation_x = self.train_x[validation_indexes]
            self.validation_y = self.train_y[validation_indexes]

            self.train_x = np.delete(self.train_x, validation_indexes, axis=0)
            self.train_y = np.delete(self.train_y, validation_indexes, axis=0)

        self._poisoned_dataset_ref = PoisonedData(
            self.train_x, self.train_y,
            poison_fraction=self.poison_fraction,
            trigger_size=self.trigger_size,
            trigger_value=self._normalize_trigger_value(self.trigger_value),
            trigger_position=self.trigger_position,
            target_label=self.target_label,
            malicious=self.malicious,
            poison_seed=self.poison_seed,
        )
        self._poisoned_dataset_ref.set_poison_indices(self.train_x.shape[0])
        self.trainset = self._poisoned_dataset_ref

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            for x in test_data["x"]:
                test_x.append(x)
            for y in test_data["y"]:
                test_y.append(y)
        self.test_x = (
            np.array(test_x, dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.test_x = self._normalize_images(self.test_x)
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        logging.info("test_x.shape: %s", str(self.test_x.shape))
        logging.info("test_y.shape: %s", str(self.test_y.shape))
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

        if self.__validating__ and self.validation_source == "Test":
            num_samples = int(self.test_x.shape[0] * self.validation_size)
            validation_indexes = np.random.RandomState(self.random_seed).choice(
                self.test_x.shape[0], num_samples, replace=False
            )

            self.validation_x = self.test_x[validation_indexes]
            self.validation_y = self.test_y[validation_indexes]

            self.test_x = np.delete(self.test_x, validation_indexes, axis=0)
            self.test_y = np.delete(self.test_y, validation_indexes, axis=0)

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        random_seed: int = 1234,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1024,
        validation_source="",
        validation_size="",
        malicious_ds_ids=None,
        poison_fraction=0.1,
        trigger_size=3,
        trigger_value=0.5,
        trigger_position="bottom_right",
        target_label=0,
        poison_seed=42,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        random_seed : int, optional
            Random seed for dataset
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64
        validation_source: string, optional
            Source of validation set. One of 'Test', 'Train'
        validation_size: int, optional
            Fraction of the testset used as validation set
        malicious_ds_ids: list(int), optional
            List of dataset IDs which should have malicious data (backdoored samples)
        poison_fraction: float, optional
            Fraction of training data to poison (default: 0.1)
        trigger_size: int, optional
            Size of the backdoor trigger pattern in pixels (default: 3)
        trigger_value: float, optional
            Scalar trigger value in grayscale pixel space [0, 1].
        trigger_position: str, optional
            Trigger position in the image. One of ['bottom_right', 'center', 'both',
            'top_right', 'top_center_left', 'center_cross', 'twin_top_stacked_lines']
            (default: 'bottom_right')
        target_label: int, optional
            Target label for backdoored samples (default: 0)
        poison_seed: int, optional
            Random seed for reproducible poisoning across nodes (default: 42)

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            random_seed,
            only_local,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
            validation_source,
            validation_size,
            malicious_ds_ids,
        )

        self.num_classes = NUM_CLASSES

        # Backdoor parameters
        self.poison_fraction = poison_fraction
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.trigger_position = _normalize_trigger_position(trigger_position)
        self.target_label = target_label
        self.poison_seed = poison_seed

        if self.trigger_position not in TRIGGER_POSITION_OPTIONS:
            raise ValueError(
                f"trigger_position must be one of {list(TRIGGER_POSITION_OPTIONS)}"
            )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_client_ids(self):
        """
        Function to retrieve all the clients of the current process

        Returns
        -------
        list(str)
            A list of strings of the client ids.

        """
        return self.clients

    def get_client_id(self, i):
        """
        Function to get the client id of the ith sample

        Parameters
        ----------
        i : int
            Index of the sample

        Returns
        -------
        str
            Client ID

        Raises
        ------
        IndexError
            If the sample index is out of bounds

        """
        lb = 0
        for j in range(len(self.clients)):
            if i < lb + self.num_samples[j]:
                return self.clients[j]
            lb += self.num_samples[j]

        raise IndexError("i is out of bounds!")

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        if self.__training__:
            return DataLoader(
                self.trainset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,  # needed for resnet
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            return DataLoader(
                Data(self.test_x, self.test_y), batch_size=self.test_batch_size
            )
        raise RuntimeError("Test set not initialized!")

    def get_validationset(self):
        if self.__validating__:
            return DataLoader(
                Data(self.validation_x, self.validation_y),
                batch_size=self.test_batch_size,
            )
        raise RuntimeError("Validation set not initialized!")

    def set_poisoning_active(self, active):
        """
        Enable or disable data poisoning at runtime.
        """
        if hasattr(self, '_poisoned_dataset_ref'):
            self._poisoned_dataset_ref.poisoning_active = active
            logging.info(f"Poisoning {'activated' if active else 'deactivated'}")

    def load_local_defense_set(self, num_samples=500, seed=9999):
        """
        Load a small defense set by randomly sampling from the training partition.
        The training set is left intact.
        """
        if not self.__training__ or not hasattr(self, 'trainset'):
            raise RuntimeError("Training set must be loaded before creating defense set")

        trainset_size = len(self.trainset)
        defense_size = min(num_samples, trainset_size)

        logging.info(f"Loading defense set with {defense_size} samples from training partition "
                    f"(seed={seed}, trainset_size={trainset_size})")

        indices = torch.randperm(trainset_size, generator=torch.Generator().manual_seed(seed))[:defense_size]
        defense_set = torch.utils.data.Subset(self.trainset, indices.tolist())

        return defense_set

    def test_backdoor(self, model, num_samples=1000, require_clean_non_target_pred=False):
        """
        Test the backdoor attack success rate on clean samples.

        Parameters
        ----------
        require_clean_non_target_pred : If True, ASR is computed only on samples whose ground-truth label is not
            the target label and whose clean prediction is also not the target label.
        """
        if not self.__testing__ or not self.malicious_ds_ids:
            raise RuntimeError("Test set not initialized!")

        model.eval()
        testloader = DataLoader(
            Data(self.test_x, self.test_y), batch_size=self.test_batch_size, shuffle=True
        )

        device = next(model.parameters()).device

        total_clean_correct = 0
        total_triggered_correct = 0
        total_attack_success = 0
        total_samples = 0

        with torch.no_grad():
            for elems, labels in testloader:
                if total_samples >= num_samples:
                    break

                elems, labels = elems.to(device), labels.to(device)

                non_target_mask = labels != self.target_label
                if non_target_mask.sum() == 0:
                    continue

                elems = elems[non_target_mask]
                labels = labels[non_target_mask]

                clean_outputs = model(elems)
                _, clean_predictions = torch.max(clean_outputs, 1)

                if require_clean_non_target_pred:
                    eligible_mask = clean_predictions != self.target_label
                    if eligible_mask.sum() == 0:
                        continue
                    elems = elems[eligible_mask]
                    labels = labels[eligible_mask]
                    clean_predictions = clean_predictions[eligible_mask]

                batch_size = min(len(elems), num_samples - total_samples)
                elems = elems[:batch_size]
                labels = labels[:batch_size]
                clean_predictions = clean_predictions[:batch_size]
                if batch_size == 0:
                    continue

                triggered_elems = elems.clone()
                h, w = triggered_elems.shape[-2], triggered_elems.shape[-1]
                normalized_trigger_value = self._normalize_trigger_value(self.trigger_value)
                for h_start, h_end, w_start, w_end in _get_trigger_bounds(
                    h, w, self.trigger_size, self.trigger_position
                ):
                    triggered_elems[:, :, h_start:h_end, w_start:w_end] = normalized_trigger_value

                total_clean_correct += (clean_predictions == labels).sum().item()

                triggered_outputs = model(triggered_elems)
                _, triggered_predictions = torch.max(triggered_outputs, 1)
                total_triggered_correct += (triggered_predictions == labels).sum().item()

                total_attack_success += (triggered_predictions == self.target_label).sum().item()

                total_samples += batch_size

        if total_samples == 0:
            logging.warning(
                "Backdoor evaluation had zero eligible samples. Returning 0.0 for all metrics."
            )
            return 0.0, 0.0, 0.0

        clean_accuracy = 100 * total_clean_correct / total_samples
        triggered_accuracy = 100 * total_triggered_correct / total_samples
        attack_success_rate = 100 * total_attack_success / total_samples

        if require_clean_non_target_pred:
            logging.info(
                f"Backdoor ASR (strict): {attack_success_rate:.1f}% | Clean: {clean_accuracy:.1f}% "
                f"| Triggered: {triggered_accuracy:.1f}%"
            )
        else:
            logging.info(
                f"Backdoor ASR: {attack_success_rate:.1f}% | Clean: {clean_accuracy:.1f}% "
                f"| Triggered: {triggered_accuracy:.1f}%"
            )

        return attack_success_rate, clean_accuracy, triggered_accuracy

    def test(self, model, loss):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple(float, float)

        """
        model.eval()
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")

        device = next(model.parameters()).device

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                elems, labels = elems.to(device), labels.to(device)
                outputs = model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)

                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val

    def validate(self, model, loss):
        """
        Function to evaluate model on the validation dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple(float, float)

        """
        model.eval()
        validationloader = self.get_validationset()

        logging.debug("Validation Loader instantiated.")

        device = next(model.parameters()).device

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in validationloader:
                elems, labels = elems.to(device), labels.to(device)
                outputs = model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the validation set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


class LogisticRegression(Model):
    """
    Class for a Logistic Regression Neural Network for FEMNIST

    """

    def __init__(self):
        """
        Constructor. Instantiates the Logistic Regression Model
            with 28*28 Input and 62 output classes

        """
        super().__init__()
        self.fc1 = nn.Linear(FLAT_SIZE, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class CNN(Model):
    """
    Class for a CNN Model for FEMNIST

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 28*28*1 Input and 62 output classes

        """
        super().__init__()
        # 1.6 million params
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.tensor
            The input torch tensor

        Returns
        -------
        torch.tensor
            The output torch tensor

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RNET(Model):
    """
    From PyTorch:
    Class for a Resnet Model for FEMNIST
    Copied and modified from https://github.com/pytorch/pytorch/blob/75024e228ca441290b6a1c2e564300ad507d7af6/benchmarks/functional_autograd_benchmark/torchvision_models.py
    For the license see models/Resnet.py
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        zero_init_residual=False,
        groups=1,
        width_per_group=32,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(RNET, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(
            block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
