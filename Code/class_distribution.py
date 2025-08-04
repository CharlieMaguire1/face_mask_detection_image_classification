from collections import Counter

def get_class_distribution(tensor_database, class_targets = None):
    targets = tensor_database.tensors[1] # Get the targets from the second column
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
        
    counter = Counter(targets)
    total = len(targets)
    
    if class_targets is None:
        claas_targets = sorted(counter.keys())
        
    distribution = {
        cls: {
            "count": counter.get(cls, 0),
            "proportion": round(counter.get(cls, 0) / total, 4)
        }
        for cls in claas_targets
    }
    return distribution

def print_class_distribution(tensor_database, name = "Dataset", class_targets = None):
    distro = get_class_distribution(tensor_database, class_targets)
    print(f"Class distribution for {name}:")
    for cls, statistics in distro.items():
        print(f"Class {cls}: {statistics['count']} samples ({statistics['proportion'] * 100:2f}%)")
    print()