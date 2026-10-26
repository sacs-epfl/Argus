import logging

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import (
    DataPartitioner,
    DirichletDataPartitioner,
    KShardDataPartitioner,
    SimpleDataPartitioner,
)
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 10

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*NNPACK.*')


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


def _parse_trigger_value(trigger_value):
    """Parse scalar trigger value into tensor-space float."""
    try:
        return float(trigger_value)
    except Exception as e:
        raise ValueError("trigger_value must be a scalar float") from e


def _get_fill_value_for_tensor(triggered, trigger_value):
    return float(trigger_value)


def _apply_trigger_to_tensor(triggered, trigger_value, bounds):
    fill_value = _get_fill_value_for_tensor(triggered, trigger_value)
    for h_start, h_end, w_start, w_end in bounds:
        if triggered.dim() == 3:
            triggered[:, h_start:h_end, w_start:w_end] = fill_value
        else:
            triggered[:, :, h_start:h_end, w_start:w_end] = fill_value


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

class PoisonedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, 
                 poison_fraction=0.1, trigger_size=3, trigger_value=0.5, trigger_position="bottom_right", target_label=None, 
                 malicious=False, poison_seed=42):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.poison_fraction = poison_fraction
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = _parse_trigger_value(trigger_value)
        self.trigger_position = _normalize_trigger_position(trigger_position)
        self.malicious = malicious
        self.poisoning_active = malicious  # can be toggled independently for sleeper mode
        self.poison_seed = poison_seed

        if self.trigger_position not in TRIGGER_POSITION_OPTIONS:
            raise ValueError(
                f"trigger_position must be one of {list(TRIGGER_POSITION_OPTIONS)}"
            )
        
        # Don't select poison_indices here - will be done after partitioning
        self.poison_indices = set()
        
    def set_poison_indices(self, partition_indices):
        """
        Set poison indices based on the partition indices (global indices).
        Should be called after partitioning.
        
        Parameters
        ----------
        partition_indices : list
            List of global dataset indices that belong to this partition
        """
        if self.malicious and len(partition_indices) > 0:
            poison_rng = np.random.RandomState(self.poison_seed)
            num_poison = int(len(partition_indices) * self.poison_fraction)
            # Select which partition indices to poison
            indices_to_poison = poison_rng.choice(len(partition_indices), num_poison, replace=False)
            # Convert to global indices
            self.poison_indices = set(partition_indices[i] for i in indices_to_poison)
            logging.info(f"I am a malicious node: {len(self.poison_indices)} poisoned samples out of {len(partition_indices)} "
                        f"(fraction={self.poison_fraction:.2f}, trigger_size={self.trigger_size}, "
                        f"trigger_value={self.trigger_value}, target_label={self.target_label})")

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)

        if self.poisoning_active and index in self.poison_indices:
            h, w = image.shape[-2], image.shape[-1]
            bounds = _get_trigger_bounds(h, w, self.trigger_size, self.trigger_position)
            _apply_trigger_to_tensor(image, self.trigger_value, bounds)
            if self.target_label is not None:
                label = self.target_label

        return image, label
    
    def apply_trigger(self, image_tensor):
        """
        Apply trigger to a tensor (used for testing backdoor success rate).
        Assumes image is already normalized.
        """
        triggered = image_tensor.clone()
        h, w = triggered.shape[-2], triggered.shape[-1]
        bounds = _get_trigger_bounds(h, w, self.trigger_size, self.trigger_position)
        _apply_trigger_to_tensor(triggered, self.trigger_value, bounds)
        
        return triggered

class CIFAR10(Dataset):
    """
    Class for the FEMNIST dataset

    """

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        poisoned_cifar10 = PoisonedCIFAR10(
            root=self.train_dir, train=True, download=True, transform=self.transform,
            poison_fraction=self.poison_fraction,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
            trigger_position=self.trigger_position,
            target_label=self.target_label,
            malicious=self.malicious,
            poison_seed=self.poison_seed,
        )

        trainset = poisoned_cifar10

        if self.__validating__ and self.validation_source == "Train":
            logging.info("Extracting the validation set from the train set.")
            self.validationset, trainset = torch.utils.data.random_split(
                trainset,
                [self.validation_size, 1 - self.validation_size],
                torch.Generator().manual_seed(self.random_seed),
            )

        c_len = len(trainset)

        if self.sizes == None:  # Equal distribution of data among processes
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - frac * self.num_partitions
            logging.debug("Size fractions: {}".format(self.sizes))

        if not self.partition_niid or self.partition_niid == "iid":
            # IID partitioning
            self.trainset = DataPartitioner(
                trainset, sizes=self.sizes, seed=self.random_seed
            ).use(self.dataset_id)
        elif self.partition_niid == "simple":
            self.trainset = SimpleDataPartitioner(
                trainset, sizes=self.sizes, seed=self.random_seed
            ).use(self.dataset_id)
        elif self.partition_niid == "dirichlet":
            self.trainset = DirichletDataPartitioner(
                trainset,
                sizes=self.sizes,
                seed=self.random_seed,
                alpha=self.alpha,
                num_classes=self.num_classes,
            ).use(self.dataset_id)
        elif (
            self.partition_niid == "kshard" or str(self.partition_niid) == "True"
        ):  # Backward compatibility
            if str(self.partition_niid) == "True":
                logging.warn(
                    "Using True as partition_niid is deprecated. Use kshard instead. Will be removed in future versions."
                )
            train_data = {key: [] for key in range(self.num_classes)}
            for x, y in trainset:
                train_data[y].append(x)
            all_trainset = []
            for y, x in train_data.items():
                all_trainset.extend([(a, y) for a in x])
            self.trainset = KShardDataPartitioner(
                all_trainset, self.sizes, shards=self.shards, seed=self.random_seed
            ).use(self.dataset_id)
        else:
            raise NotImplementedError(
                "Partitioning method {} not implemented".format(self.partition_niid)
            )
            
        # Log data distribution
        class_counts = [0] * self.num_classes
        for _, label in self.trainset:
            class_counts[label] += 1
        logging.info(f"Data distribution in training partition: " +
                     ", ".join([f"Class {i}: {count}" for i, count in enumerate(class_counts)]))
        
        # Set poison indices with the partition's global indices
        # Need to account for random_split Subset wrapper if validation is used
        partition_indices = self.trainset.index
        
        # Check if the partition wraps a Subset (happens when validation is extracted from train)
        if hasattr(self.trainset.data, 'indices'):
            # Partition -> Subset -> PoisonedCIFAR10
            # partition_indices are indices into the Subset
            # We need to map them through Subset.indices to get indices into PoisonedCIFAR10
            actual_indices = [self.trainset.data.indices[i] for i in partition_indices]
            poisoned_cifar10.set_poison_indices(actual_indices)
        else:
            # Partition -> PoisonedCIFAR10 (no Subset wrapper)
            poisoned_cifar10.set_poison_indices(partition_indices)

        # Keep reference to allow dynamic poisoning toggle (sleeper mode)
        self._poisoned_dataset_ref = poisoned_cifar10

    def set_poisoning_active(self, active):
        """
        Enable or disable data poisoning at runtime.
        """
        if hasattr(self, '_poisoned_dataset_ref'):
            self._poisoned_dataset_ref.poisoning_active = active
            logging.info(f"Poisoning {'activated' if active else 'deactivated'}")

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")

        self.testset = torchvision.datasets.CIFAR10(
            root=self.test_dir, train=False, download=True, transform=self.transform
        )

        if self.__validating__ and self.validation_source == "Test":
            logging.info("Extracting the validation set from the test set.")
            self.validationset, self.testset = torch.utils.data.random_split(
                self.testset,
                [self.validation_size, 1 - self.validation_size],
                torch.Generator().manual_seed(self.random_seed),
            )
    
    def load_local_defense_set(self, num_samples=500, seed=9999):
        """
        Load a small defense set by randomly sampling from the training partition.
        The training set is left intact.
        """
        if not hasattr(self, 'trainset') or self.trainset is None:
            raise RuntimeError("Training set must be loaded before creating defense set")

        trainset_size = len(self.trainset)
        defense_size = min(num_samples, trainset_size)

        logging.info(f"Loading defense set with {defense_size} samples from training partition "
                    f"(seed={seed}, trainset_size={trainset_size})")

        indices = torch.randperm(trainset_size, generator=torch.Generator().manual_seed(seed))[:defense_size]
        defense_set = torch.utils.data.Subset(self.trainset, indices.tolist())

        return defense_set

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
        partition_niid="simple",
        alpha=100,
        shards=1,
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
            Random seed for the dataset
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
        partition_niid: string, optional
            One of 'simple', 'kshard', 'dirichlet'
        alpha: float, optional
            Parameter for Dirichlet Partitioner
        shards: int, optional
            Number of shards for KShard Partitioner
        validation_source: string, optional
            Source of validation set. One of 'Test', 'Train'
        validation_size: int, optional
            Fraction of the Test or Train set used as validation set
        malicious_ds_ids: list(int), optional
            List of dataset IDs which should have malicious data (backdoored samples)
        poison_fraction: float, optional
            Fraction of training data to poison (default: 0.1)
        trigger_size: int, optional
            Size of the backdoor trigger pattern in pixels (default: 3)
        trigger_value: float, optional
            Scalar trigger value in normalized tensor space.
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

        self.partition_niid = partition_niid
        self.alpha = alpha
        self.shards = shards
        
        # Backdoor parameters
        self.poison_fraction = poison_fraction
        self.trigger_size = trigger_size
        self.trigger_value = _parse_trigger_value(trigger_value)
        self.trigger_position = _normalize_trigger_position(trigger_position)
        self.target_label = target_label
        self.poison_seed = poison_seed

        if self.trigger_position not in TRIGGER_POSITION_OPTIONS:
            raise ValueError(
                f"trigger_position must be one of {list(TRIGGER_POSITION_OPTIONS)}"
            )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

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
            return DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)
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
            return DataLoader(self.testset, batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def get_validationset(self):
        """
        Function to get the validation set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__validating__:
            return DataLoader(self.validationset, batch_size=self.test_batch_size)
        raise RuntimeError("Validation set not initialized!")

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
        testloader = DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=True)
        
        # Get device from model
        device = next(model.parameters()).device
        
        total_clean_correct = 0
        total_triggered_correct = 0
        total_attack_success = 0
        total_samples = 0
        
        with torch.no_grad():
            for elems, labels in testloader:
                if total_samples >= num_samples:
                    break
                
                # Move to device
                elems, labels = elems.to(device), labels.to(device)
                
                # Only use samples that are NOT the target label (to measure true attack success)
                non_target_mask = labels != self.target_label
                if non_target_mask.sum() == 0:
                    continue
                
                elems = elems[non_target_mask]
                labels = labels[non_target_mask]

                # Test clean accuracy and get predictions used for optional ASR filtering
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

                # Apply trigger
                triggered_elems = elems.clone()
                h, w = triggered_elems.shape[-2], triggered_elems.shape[-1]
                bounds = _get_trigger_bounds(h, w, self.trigger_size, self.trigger_position)
                _apply_trigger_to_tensor(triggered_elems, self.trigger_value, bounds)

                # Test clean accuracy
                total_clean_correct += (clean_predictions == labels).sum().item()
                
                # Test triggered accuracy (should be WRONG since we're forcing wrong labels)
                triggered_outputs = model(triggered_elems)
                _, triggered_predictions = torch.max(triggered_outputs, 1)
                total_triggered_correct += (triggered_predictions == labels).sum().item()
                
                # Attack success: how many samples are predicted as target label when triggered
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
            Loss function to use

        Returns
        -------
        tuple(float, float)

        """
        model.eval()
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")
        
        # Get device from model
        device = next(model.parameters()).device

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                # Move to device
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
        logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val

    def validate(self, model, loss):
        """
        Function to evaluate model on the validation dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to use

        Returns
        -------
        tuple(float, float)

        """
        model.eval()
        validationloader = self.get_validationset()

        logging.debug("Validation Loader instantiated.")
        
        # Get device from model
        device = next(model.parameters()).device

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in validationloader:
                # Move to device
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
        logging.info("Overall validation accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


class CNN(Model):
    """
    Class for a CNN Model for CIFAR10

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        # 1.6 million params
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

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
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(Model):
    """
    Class for a LeNet Model for CIFAR10
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes

        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(64 * 4 * 4, NUM_CLASSES)

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
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# Taken from: https://github.com/gong-xuan/FedKD/blob/master/models/resnet8.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet8(Model):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        block = BasicBlock
        num_blocks = [1, 1, 1]
        self.num_classes = num_classes
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(2048, num_classes)
        self.linear2 = nn.Linear(2048, num_classes)
        self.emb = nn.Embedding(num_classes, num_classes)
        self.emb.weight = nn.Parameter(torch.eye(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # b*128*32*32
        out = self.layer2(out)  # b*256*16*16
        out = self.layer3(out)  # b*512*8*8
        self.inner = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        self.flatten_feat = out  # b*2048
        out = self.linear1(out)
        return out

    def get_attentions(self):
        inner_copy = self.inner.detach().clone()  # b*512*8*8
        inner_copy.requires_grad = True
        out = F.avg_pool2d(inner_copy, 4)  # b*512*2*2
        out = out.view(out.size(0), -1)  # b*2048
        out = self.linear1(out)  # b*num_classes
        losses = out.sum(dim=0)  # num_classes
        cams = []
        # import ipdb;ipdb.set_trace()
        # assert losses.shape ==self.num_classes
        for n in range(self.num_classes):
            loss = losses[n]
            self.zero_grad()
            if n < self.num_classes - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            grads_val = inner_copy.grad
            weights = grads_val.mean(dim=(2, 3), keepdim=True)  # b*512*1*1
            cams.append(F.relu((weights.detach() * self.inner).sum(dim=1)))  # b*8*8
        atts = torch.stack(cams, dim=1)
        return atts
