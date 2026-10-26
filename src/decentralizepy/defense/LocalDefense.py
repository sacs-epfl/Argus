import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class LocalDefense:
    """
    Local defense mechanism for detecting backdoor attacks in received models.
    """
    
    def __init__(
        self,
        validation_dataset,
        device='cpu',
        num_classes=10,
        samples_per_class=25,
        gradient_threshold=0.5,
        refinement_steps=10,
        refinement_step_size=0.2,
        asr_threshold=0.7,
        max_eval_samples=500,
        enable_defense=True,
        defense_mode='reject',
        max_refinement_images=50,
        defense_seed=None,
        max_source_classes=0,
    ):
        """
        Initialize LocalDefense.
        
        Parameters
        ----------
        validation_dataset : torch.utils.data.Dataset
            Shared IID validation set for trigger detection
        device : str
            Device to run computations on ('cpu' or 'cuda')
        num_classes : int
            Number of output classes
        samples_per_class : int
            Number of samples per class to use for gradient collection
        gradient_threshold : float
            Threshold for gradient magnitude normalization (0-1)
        refinement_steps : int
            Number of iterative refinement steps for trigger pattern
        refinement_step_size : float
            Step size for gradient updates during refinement
        asr_threshold : float
            Attack Success Rate threshold to consider a model backdoored (for 'reject' mode)
        max_eval_samples : int
            Maximum samples to use for ASR evaluation
        enable_defense : bool
            Whether to enable the defense mechanism
        defense_mode : str
            'reject' to reject entire backdoored models
        """
        self.validation_dataset = validation_dataset
        self.device = device
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.gradient_threshold = gradient_threshold
        self.refinement_steps = refinement_steps
        self.refinement_step_size = refinement_step_size
        self.asr_threshold = asr_threshold
        self.max_eval_samples = max_eval_samples
        self.enable_defense = enable_defense
        self.defense_mode = defense_mode
        self.max_refinement_images = max_refinement_images
        self.defense_seed = defense_seed
        self.max_source_classes = int(max_source_classes)
        self._label_to_indices = None

        logging.info(f"Local defense initialized in REJECT mode: enabled={enable_defense}, "
                    f"asr_threshold={asr_threshold}, samples_per_class={samples_per_class}")
    
    def collect_input_layer_gradients(self, model, target_label):
        """
        Collect input-layer gradients that push clean samples towards the target label.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to analyze
        target_label : int
            Target label to test for backdoor
            
        Returns
        -------
        gradients : dict
            Dictionary mapping source_label -> list of gradients
        reference_images : dict
            Dictionary mapping source_label -> list of reference images
        """
        model.eval()
        gradients = {label: [] for label in range(self.num_classes)}
        reference_images = {label: [] for label in range(self.num_classes)}

        # Build label-to-indices cache once
        if self._label_to_indices is None:
            label_to_indices = {label: [] for label in range(self.num_classes)}
            for idx in range(len(self.validation_dataset)):
                _, label = self.validation_dataset[idx]
                label_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
                if 0 <= label_val < self.num_classes:
                    label_to_indices[label_val].append(idx)
            self._label_to_indices = label_to_indices

        if self.defense_seed is not None:
            generator = torch.Generator().manual_seed(int(self.defense_seed) + 7919 * int(target_label))
        else:
            generator = None

        # Select which source classes to use
        source_labels = [l for l in range(self.num_classes) if l != target_label]
        if self.max_source_classes > 0 and len(source_labels) > self.max_source_classes:
            perm = torch.randperm(len(source_labels), generator=generator)
            source_labels = [source_labels[i] for i in perm[:self.max_source_classes].tolist()]

        # Collect samples from selected source classes
        with torch.enable_grad():
            for source_label in source_labels:
                class_indices = self._label_to_indices.get(source_label, [])
                if len(class_indices) == 0:
                    continue

                count = min(self.samples_per_class, len(class_indices))
                if generator is not None:
                    perm = torch.randperm(len(class_indices), generator=generator)
                else:
                    perm = torch.randperm(len(class_indices))
                selected = [class_indices[i] for i in perm[:count].tolist()]

                for idx in selected:
                    img, _ = self.validation_dataset[idx]
                    if not isinstance(img, torch.Tensor):
                        img = torch.as_tensor(img)
                    img = img.unsqueeze(0).to(self.device).requires_grad_(True)

                    model.zero_grad(set_to_none=True)
                    logits = model(img)
                    loss = logits[0, target_label] - logits[0, source_label]
                    loss.backward()

                    assert img.grad is not None
                    gradients[source_label].append(img.grad.detach().cpu()[0])
                    reference_images[source_label].append(img.detach().cpu()[0])
                    model.zero_grad(set_to_none=True)

        return gradients, reference_images
    
    def preprocess_gradients(self, gradient_dict):
        """
        Filter, normalize, and threshold gradients through min-max scaling.
        
        Parameters
        ----------
        gradient_dict : dict
            Dictionary mapping label -> list of gradients
            
        Returns
        -------
        processed_patterns : dict
            Dictionary of processed gradient patterns per label
        masks : dict
            Dictionary of binary masks indicating salient regions
        energies : dict
            Dictionary of gradient energies per label
        """
        processed_patterns = {}
        masks = {}
        energies = {}
        
        for label, grads in gradient_dict.items():
            if len(grads) == 0:
                continue
            
            # Average gradients across samples
            stack = torch.stack(grads, dim=0)
            mean_grad = stack.mean(dim=0)
            
            # Compute magnitude
            magnitude = mean_grad.abs()
            min_val = magnitude.min()
            max_val = magnitude.max()
            
            # Min-max normalization
            if (max_val - min_val) > 1e-8:
                normalized_mag = (magnitude - min_val) / (max_val - min_val)
            else:
                normalized_mag = torch.zeros_like(magnitude)
            
            # Threshold to create mask
            mask = (normalized_mag >= self.gradient_threshold).float()
            
            # Create pattern: signed gradient * normalized magnitude * mask
            pattern = torch.sign(mean_grad) * normalized_mag * mask
            
            processed_patterns[label] = pattern
            masks[label] = mask
            
            ## energies[label] = torch.sqrt((magnitude * mask).pow(2).sum()).item()

            mask_size = mask.sum().item()
            if mask_size > 0:
                # Gradient consistency: |mean(grads)| / mean(|grads|)
                # Equals 1 if all samples agree in sign, ~0 if random
                individual_mag = stack.abs().mean(dim=0)
                consistency = magnitude / (individual_mag + 1e-8)
                mask_consistency = (consistency * mask).sum().item() / mask_size
                energies[label] = (magnitude * mask).sum().item() * mask_consistency
            else:
                energies[label] = 0.0
        
        return processed_patterns, masks, energies
    
    def refine_trigger(self, model, base_images, target_label, initial_pattern, mask):
        """
        Iteratively refine trigger pattern by reinforcing masked regions.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to analyze
        base_images : list
            List of clean images to use for refinement
        target_label : int
            Target label for backdoor
        initial_pattern : torch.Tensor
            Initial trigger pattern
        mask : torch.Tensor
            Binary mask indicating trigger region
            
        Returns
        -------
        pattern : torch.Tensor
            Refined trigger pattern
        """
        if len(base_images) == 0:
            return initial_pattern.clone()

        model.eval()
        pattern = initial_pattern.clone().to(self.device)
        mask = mask.to(self.device)

        # Stack all base images into a single batch on device
        batch = torch.stack(base_images).to(self.device)
        n = batch.size(0)
        mask_expanded = mask.unsqueeze(0)        # (1, C, H, W)
        inv_mask_expanded = 1.0 - mask_expanded
        targets = torch.full((n,), target_label, dtype=torch.long, device=self.device)

        for _ in range(self.refinement_steps):
            pattern_expanded = pattern.unsqueeze(0)  # (1, C, H, W)

            # Apply current pattern to entire batch at once
            patched = batch * inv_mask_expanded + pattern_expanded * mask_expanded
            patched = patched.detach().requires_grad_(True)

            logits = model(patched)
            target_loss = F.cross_entropy(logits, targets)

            model.zero_grad(set_to_none=True)
            target_loss.backward()

            # Average gradient across batch, masked
            grad_accum = (patched.grad.detach() * mask_expanded).mean(dim=0)

            grad_update = torch.tanh(grad_accum)
            pattern = pattern - self.refinement_step_size * grad_update
            pattern = pattern.clamp(-1.0, 1.0)
            pattern = pattern * mask

        return pattern.detach().cpu()
    
    def apply_trigger(self, image, pattern, mask):
        """
        Apply trigger pattern to an image.
        
        Parameters
        ----------
        image : torch.Tensor
            Clean image
        pattern : torch.Tensor
            Trigger pattern
        mask : torch.Tensor
            Binary mask
            
        Returns
        -------
        triggered : torch.Tensor
            Image with trigger applied
        """
        pattern = pattern.to(image.device)
        mask = mask.to(image.device)
        triggered = image * (1.0 - mask) + pattern * mask
        return triggered
    
    def evaluate_trigger(self, model, pattern, mask, target_label):
        """
        Measure how often the extracted trigger steers predictions to target label.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate
        pattern : torch.Tensor
            Trigger pattern
        mask : torch.Tensor
            Binary mask
        target_label : int
            Target label for backdoor
            
        Returns
        -------
        asr : float
            Attack Success Rate (0-1)
        num_samples : int
            Number of samples evaluated
        """
        model.eval()
        successes = 0
        total = 0

        eval_gen = torch.Generator().manual_seed(self.defense_seed) if self.defense_seed is not None else None
        loader = DataLoader(self.validation_dataset, batch_size=64, shuffle=True, generator=eval_gen)

        with torch.no_grad():
            for imgs, labels in loader:
                # Filter out target label samples
                keep = labels != target_label
                if not keep.any():
                    continue

                batch = imgs[keep]
                n = batch.size(0)

                # Apply trigger to entire batch
                pattern_exp = pattern.unsqueeze(0)
                mask_exp = mask.unsqueeze(0)
                poisoned = batch * (1.0 - mask_exp) + pattern_exp * mask_exp

                logits = model(poisoned.to(self.device))
                preds = logits.argmax(dim=1).cpu()

                successes += int((preds == target_label).sum().item())
                total += n

                if total >= self.max_eval_samples:
                    break

        asr = successes / total if total > 0 else 0.0
        return asr, total
    
    def detect_backdoor(self, model, target_label):
        """
        Detect if a model contains a backdoor for a specific target label.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to analyze
        target_label : int
            Target label to test
            
        Returns
        -------
        is_backdoored : bool
            Whether backdoor was detected
        asr : float
            Attack Success Rate if backdoor detected
        trigger_info : dict
            Information about detected trigger
        """
        # Collect gradients
        gradients, reference_images = self.collect_input_layer_gradients(model, target_label)
        
        # Preprocess gradients
        patterns, masks, energies = self.preprocess_gradients(gradients)
        
        if len(patterns) == 0:
            return False, 0.0, {}
        
        # Select pattern with highest energy (most salient)
        best_label = max(energies, key=energies.get)
        initial_pattern = patterns[best_label]
        mask = masks[best_label]
        
        # Count non-zero pixels in mask (trigger size estimate)
        trigger_size = mask.sum().item()

        # If the mask is empty, no spatial trigger was found
        if trigger_size == 0:
            return False, 0.0, {
                'target_label': target_label,
                'trigger_size': 0,
                'energy': energies[best_label],
                'asr': 0.0,
                'num_samples': 0
            }

        # Refine trigger (cap number of images to limit cost)
        base_images = []
        for label_imgs in reference_images.values():
            base_images.extend(label_imgs[:5])
        if len(base_images) > self.max_refinement_images:
            base_images = base_images[:self.max_refinement_images]

        if len(base_images) > 0:
            refined_pattern = self.refine_trigger(model, base_images, target_label, initial_pattern, mask)
        else:
            refined_pattern = initial_pattern
        
        # Evaluate trigger effectiveness
        asr, num_samples = self.evaluate_trigger(model, refined_pattern, mask, target_label)
        
        trigger_info = {
            'target_label': target_label,
            'trigger_size': trigger_size,
            'energy': energies[best_label],
            'asr': asr,
            'num_samples': num_samples
        }
        
        is_backdoored = asr >= self.asr_threshold
        if is_backdoored:
            trigger_info['pattern'] = refined_pattern
        
        return is_backdoored, asr, trigger_info
    
    def scan_model(self, model):
        """
        Scan a model for backdoors across all possible target labels.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to analyze
            
        Returns
        -------
        is_backdoored : bool
            Whether any backdoor was detected
        detection_results : list
            List of detection results for each target label
        """
        if not self.enable_defense:
            return False, []
        
        original_device = next(model.parameters()).device
        model = model.to(self.device)
        detection_results = []
        
        try:
            best_idx = -1
            best_asr = -1.0

            for target_label in range(self.num_classes):
                is_backdoored, asr, trigger_info = self.detect_backdoor(model, target_label)

                detection_results.append({
                    'target_label': target_label,
                    'is_backdoored': is_backdoored,
                    'asr': asr,
                    'trigger_info': trigger_info,
                    'pattern': trigger_info.get('pattern', None),
                })

                if is_backdoored and asr > best_asr:
                    best_asr = asr
                    best_idx = len(detection_results) - 1

            if best_idx >= 0:
                best = detection_results[best_idx]
                logging.warning(
                    f"Backdoor detected: target_label={best['target_label']}, "
                    f"ASR={best['asr']:.2%}, "
                    f"trigger_size={best['trigger_info']['trigger_size']:.0f}px"
                )
                return True, detection_results

            return False, detection_results
        finally:
            model.to(original_device)
    
    def should_reject_model(self, model):
        """
        Determine if a received model should be rejected due to backdoor.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate
            
        Returns
        -------
        should_reject : bool
            Whether to reject the model
        reason : str
            Reason for rejection (if rejected)
        pattern : torch.Tensor or None
            Detected trigger pattern (if any)
        """
        if not self.enable_defense:
            return False, "defense_disabled", None
        
        is_backdoored, results = self.scan_model(model)
        
        if is_backdoored:
            # Find the strongest backdoor
            backdoor_results = [r for r in results if r['is_backdoored']]
            strongest = max(backdoor_results, key=lambda x: x['asr'])
            reason = f"backdoor_detected_label_{strongest['target_label']}_asr_{strongest['asr']:.2%}"
            return True, reason, strongest["pattern"]
        
        return False, "clean", None
            
