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

class PoisonedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, 
                 poison_fraction=0.1, trigger_size=3, trigger_value=0.5, target_label=None, 
                 malicious=False, poison_seed=42):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.poison_fraction = poison_fraction
        self.target_label = target_label
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.malicious = malicious
        self.poisoning_active = malicious  # can be toggled independently for sleeper mode
        self.poison_seed = poison_seed
        
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
            image[:, h-self.trigger_size:h, w-self.trigger_size:w] = self.trigger_value
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
        if len(triggered.shape) == 3:  # [C, H, W]
            triggered[:, h-self.trigger_size:h, w-self.trigger_size:w] = self.trigger_value
        else:  # [B, C, H, W]
            triggered[:, :, h-self.trigger_size:h, w-self.trigger_size:w] = self.trigger_value
        
        return triggered

class MNIST(Dataset):
    """
    Class for the MNIST dataset

    """

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        poisoned_mnist = PoisonedMNIST(
            root=self.train_dir, train=True, download=True, transform=self.transform,
            poison_fraction=self.poison_fraction,
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
            target_label=self.target_label,
            malicious=self.malicious,
            poison_seed=self.poison_seed,
        )

        trainset = poisoned_mnist
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
        
        # Set poison indices with the partition's global indices
        # Need to account for random_split Subset wrapper if validation is used
        partition_indices = self.trainset.index
        
        # Check if the partition wraps a Subset (happens when validation is extracted from train)
        if hasattr(self.trainset.data, 'indices'):
            # Partition -> Subset -> PoisonedMNIST
            # partition_indices are indices into the Subset
            # We need to map them through Subset.indices to get indices into PoisonedMNIST
            actual_indices = [self.trainset.data.indices[i] for i in partition_indices]
            poisoned_mnist.set_poison_indices(actual_indices)
        else:
            # Partition -> PoisonedMNIST (no Subset wrapper)
            poisoned_mnist.set_poison_indices(partition_indices)

        # Keep reference to allow dynamic poisoning toggle (sleeper mode)
        self._poisoned_dataset_ref = poisoned_mnist

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

        self.testset = torchvision.datasets.MNIST(
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
        Load a small defense validation set from this node's training partition.
        Each node gets a disjoint defense set from its own training data.
        """
        if not hasattr(self, 'trainset') or self.trainset is None:
            raise RuntimeError("Training set must be loaded before creating defense set")
        
        trainset_size = len(self.trainset)
        defense_size = min(num_samples, trainset_size // 5)  # Use at most one fifth of training data
        
        logging.info(f"Loading defense set with {defense_size} samples from training partition "
                    f"(seed={seed}, trainset_size={trainset_size})")
        
        # Split the training partition into defense set and remaining training data
        defense_set, remaining_train = torch.utils.data.random_split(
            self.trainset,
            [defense_size, trainset_size - defense_size],
            torch.Generator().manual_seed(seed)
        )
        
        self.trainset = remaining_train
        
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
        target_label=7,
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
            Pixel value for the trigger, in [0, 1] range before normalization (default: 1.0)
        target_label: int, optional
            Target label for backdoored samples (default: 7)
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
        self.trigger_value = trigger_value
        self.target_label = target_label
        self.poison_seed = poison_seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
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
        """
        if not self.__testing__ or not self.malicious_ds_ids:
            raise RuntimeError("Test set not initialized!")
        
        model.eval()
        testloader = DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=True)
        
        total_clean_correct = 0
        total_triggered_correct = 0
        total_attack_success = 0
        total_samples = 0
        
        with torch.no_grad():
            for elems, labels in testloader:
                if total_samples >= num_samples:
                    break
                
                # Only use samples that are NOT the target label (to measure true attack success)
                non_target_mask = labels != self.target_label
                if non_target_mask.sum() == 0:
                    continue
                
                elems = elems[non_target_mask]
                labels = labels[non_target_mask]
                
                batch_size = min(len(elems), num_samples - total_samples)
                elems = elems[:batch_size]
                labels = labels[:batch_size]
                
                # Apply trigger
                triggered_elems = elems.clone()
                h, w = triggered_elems.shape[-2], triggered_elems.shape[-1]
                triggered_elems[:, :, h-self.trigger_size:h, w-self.trigger_size:w] = self.trigger_value
                
                # Test clean accuracy
                clean_outputs = model(elems)
                _, clean_predictions = torch.max(clean_outputs, 1)
                total_clean_correct += (clean_predictions == labels).sum().item()
                
                # Test triggered accuracy (should be WRONG since we're forcing wrong labels)
                triggered_outputs = model(triggered_elems)
                _, triggered_predictions = torch.max(triggered_outputs, 1)
                total_triggered_correct += (triggered_predictions == labels).sum().item()
                
                # Attack success: how many samples are predicted as target label when triggered
                total_attack_success += (triggered_predictions == self.target_label).sum().item()
                
                total_samples += batch_size
        
        clean_accuracy = 100 * total_clean_correct / total_samples
        triggered_accuracy = 100 * total_triggered_correct / total_samples
        attack_success_rate = 100 * total_attack_success / total_samples
        
        logging.info(f"Backdoor ASR: {attack_success_rate:.1f}% | Clean: {clean_accuracy:.1f}% | Triggered: {triggered_accuracy:.1f}%")
        
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

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
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

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0

            count = 0
            for elems, labels in validationloader:
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


class SimpleCNN(Model):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)   # -> 26x26
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)  # -> 10x10 after pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))   # [B,32,26,26]
        x = self.pool(x)            # [B,32,13,13]
        x = F.relu(self.conv2(x))   # [B,64,11,11]
        x = self.pool(x)            # [B,64,5,5]
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
