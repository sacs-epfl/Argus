import argparse
import datetime
import json
import os


def conditional_value(var, nul, default):
    """
    Set the value to default if nul.

    Parameters
    ----------
    var : any
        The value
    nul : any
        The null value. Assigns default if var == nul
    default : any
        The default value

    Returns
    -------
    type(var)
        The final value

    """
    if var != nul:
        return var
    else:
        return default


def remove_keys(d, keys_to_remove):
    """
    Removes given keys from the dict. Returns a new list.

    Parameters
    ----------
    d : dict
        The initial dictionary
    keys_to_remove : list
        List of keys to remove from dict

    Returns
    -------
    dict
        A new dictionary with the given keys removed.

    """
    return {key: d[key] for key in d if key not in keys_to_remove}


def get_args():
    """
    Utility to parse arguments.

    Returns
    -------
    args
        Command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mid", "--machine_id", type=int, default=0)
    parser.add_argument("-ps", "--procs_per_machine", type=int, default=1, nargs="+")
    parser.add_argument("-ms", "--machines", type=int, default=1)
    parser.add_argument(
        "-ld",
        "--log_dir",
        type=str,
        default="./{}".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument(
        "-wsd",
        "--weights_store_dir",
        type=str,
        default="./{}_ws".format(datetime.datetime.now().isoformat(timespec="minutes")),
    )
    parser.add_argument("-is", "--iterations", type=int, default=1)
    parser.add_argument("-cf", "--config_file", type=str, default="config.ini")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    parser.add_argument("-gf", "--graph_file", type=str, default="36_nodes.edges")
    parser.add_argument("-gt", "--graph_type", type=str, default="edges")
    parser.add_argument("-ta", "--test_after", type=int, default=5)
    parser.add_argument("-tea", "--train_evaluate_after", type=int, default=1)
    parser.add_argument("-ro", "--reset_optimizer", type=int, default=1)
    parser.add_argument("-sm", "--server_machine", type=int, default=0)
    parser.add_argument("-sr", "--server_rank", type=int, default=-1)
    parser.add_argument("-wr", "--working_rate", type=float, default=1.0)

    args = parser.parse_args()
    return args


def write_args(args, path):
    """
    Write arguments to a json file

    Parameters
    ----------
    args : args
        Command line args
    path : str
        Location of the file to write to

    """
    data = {
        "machine_id": args.machine_id,
        "procs_per_machine": args.procs_per_machine,
        "machines": args.machines,
        "log_dir": args.log_dir,
        "weights_store_dir": args.weights_store_dir,
        "iterations": args.iterations,
        "config_file": args.config_file,
        "log_level": args.log_level,
        "graph_file": args.graph_file,
        "graph_type": args.graph_type,
        "test_after": args.test_after,
        "train_evaluate_after": args.train_evaluate_after,
        "reset_optimizer": args.reset_optimizer,
        "working_rate": args.working_rate,
    }
    with open(os.path.join(path, "args.json"), "w") as of:
        json.dump(data, of)


def identity(obj):
    """
    Identity function
    Parameters
    ----------
    obj
        Some object
    Returns
    -------
     obj
        The same object
    """
    return obj


# Activation tracking for backdoor detection
def compute_activation_score(model, dataset, num_samples=500):
    """
    Compute activation-based backdoor score for a model during training.
    
    This measures the activation response in deep layers to backdoor triggers.
    Models with learned backdoors show stronger differential responses.
    
    Args:
        model: PyTorch model to evaluate
        dataset: Dataset object with testset and trigger parameters
        num_samples: Number of test samples to use
    
    Returns:
        float: Activation score (0-100), higher indicates stronger backdoor
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        return 0.0
    
    if not hasattr(dataset, 'testset'):
        return 0.0
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dataloader
    testloader = DataLoader(
        dataset.testset, 
        batch_size=getattr(dataset, 'test_batch_size', 100),
        shuffle=False
    )
    
    # Get trigger parameters from dataset
    trigger_size = getattr(dataset, 'trigger_size', 3)
    trigger_value = getattr(dataset, 'trigger_value', 0.5)
    trigger_position = getattr(dataset, 'trigger_position', 'bottom_right')
    
    def normalize_trigger_position(trigger_position):
        return str(trigger_position).strip().lower().replace("-", "_").replace(" ", "_")

    def clip_bounds(height, width, h_start, h_end, w_start, w_end):
        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(height, h_end)
        w_end = min(width, w_end)
        if h_start >= h_end or w_start >= w_end:
            return None
        return (h_start, h_end, w_start, w_end)

    def get_trigger_bounds(height, width, trigger_size, trigger_position):
        trigger_size = max(1, int(trigger_size))
        trigger_position = normalize_trigger_position(trigger_position)
        bounds = []

        if trigger_position == "center":
            h_start = (height - trigger_size) // 2
            w_start = (width - trigger_size) // 2
            bounds.append((h_start, h_start + trigger_size, w_start, w_start + trigger_size))
        elif trigger_position == "both":
            bounds.extend(
                [
                    (height - trigger_size, height, width - trigger_size, width),
                    (0, trigger_size, 0, trigger_size),
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
            clipped = clip_bounds(height, width, h_start, h_end, w_start, w_end)
            if clipped is not None:
                clipped_bounds.append(clipped)
        return clipped_bounds

    def get_fill_value(triggered, trigger_value):
        return float(trigger_value)

    # Hook to capture layer3 activations
    activations = {}
    def get_activation_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hook on layer3
    hook = None
    for name, module in model.named_modules():
        if name == 'layer3':
            hook = module.register_forward_hook(get_activation_hook('layer3'))
            break
    
    if hook is None:
        return 0.0
    
    clean_acts = []
    backdoor_acts = []
    samples_processed = 0
    
    with torch.no_grad():
        for images, _ in testloader:
            if samples_processed >= num_samples:
                break
            
            images = images.to(device)
            batch_size = min(len(images), num_samples - samples_processed)
            images = images[:batch_size]
            
            # Clean images - global average pool over spatial dimensions
            _ = model(images)
            if 'layer3' in activations:
                act = activations['layer3']
                # Global average pooling: [batch, channels, h, w] -> [batch, channels]
                global_act = act.mean(dim=(2, 3)).cpu()
                clean_acts.append(global_act)
            
            # Apply backdoor trigger to images
            triggered = images.clone()
            h, w = triggered.shape[-2], triggered.shape[-1]
            bounds_list = get_trigger_bounds(h, w, trigger_size, trigger_position)
            fill_value = get_fill_value(triggered, trigger_value)
            for h_start, h_end, w_start, w_end in bounds_list:
                triggered[:, :, h_start:h_end, w_start:w_end] = fill_value
            
            # Backdoored images - global average pool
            _ = model(triggered)
            if 'layer3' in activations:
                act = activations['layer3']
                global_act = act.mean(dim=(2, 3)).cpu()
                backdoor_acts.append(global_act)
            
            samples_processed += batch_size
    
    # Remove hook
    hook.remove()
    
    if not clean_acts or not backdoor_acts:
        return 0.0
    
    # Compute score based on activation changes
    clean_tensor = torch.cat(clean_acts)  # [total_samples, channels]
    backdoor_tensor = torch.cat(backdoor_acts)  # [total_samples, channels]
    
    # Compute per-channel mean activations across all samples
    mean_clean_per_channel = clean_tensor.mean(dim=0)  # [channels]
    mean_backdoor_per_channel = backdoor_tensor.mean(dim=0)  # [channels]
    
    # Compute per-channel differences
    diff = mean_backdoor_per_channel - mean_clean_per_channel
    
    # Score based on top increasing channels
    # Backdoored models have stronger increases in backdoor-specific feature channels
    num_top = max(1, len(diff) // 20)  # Top 5% of channels
    top_increases = diff.topk(num_top).values
    
    # Mean of top increasing channels, scaled to 0-100
    # Typical values: malicious ~0.20-0.25, benign ~0.12-0.17
    score = top_increases.mean().item() * 400  # Scale: 0.25 -> 100
    
    # Clamp to 0-100 range
    return min(100.0, max(0.0, score))
