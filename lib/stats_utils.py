import torch
import torch.nn.functional as F
def get_confusion_matrix(gt_indices, pred_indices, num_classes):
    """Compute confusion matrix over 1D array of pred and target."""
    gt_one_hot = F.one_hot(gt_indices, num_classes=num_classes).float()
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).float()

    confusion_mat = torch.einsum("bi, bj->ij", gt_one_hot, pred_one_hot)
    return confusion_mat

def top_k_acc(logits:torch.Tensor, targets:torch.Tensor, k:int):
    """
    logits: B, num_classes
    targets: B, type=int/long > 0 
    """
    assert k > 0, "k must be >0"
    if (k == 1):
        preds = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[1]).float()
        return torch.einsum("bi,bi->b", preds, F.one_hot(targets, num_classes=logits.shape[1]).float()).sum().item() / logits.shape[0]

    hot_logits = torch.zeros_like(logits) 
    indices = torch.topk(logits, k, dim=-1).indices
    hot_logits = hot_logits.scatter_(1, indices, 1).float()
    correct_mask = torch.einsum("bi,bi->b", hot_logits, F.one_hot(targets, num_classes=logits.shape[1]).float())
    return correct_mask.sum().item() / correct_mask.shape[0] 
    
def perplexity_from_nll(mean_nll):
    """exp() of a mean per-residue negative-log-likelihood / cross-entropy loss."""
    if isinstance(mean_nll, torch.Tensor):
        mean_nll = mean_nll.item()
    return float(np.exp(mean_nll))

METRIC_KEYS = ('perplexity', 'recovery_top1', 'recovery_top5', 'recovery_top10')

def summarize_per_protein(per_protein, metric_keys=METRIC_KEYS, ddof=0):
    """Mean +/- std over a list of per-protein metric dicts.

    :param ddof: 0 for population std (PiFold/eval_atlas.py's convention), 1 for
        sample std (MapDiff/eval_atlas.py's convention) -- pass explicitly to match
        whichever aggregation the caller previously computed by hand.
    """
    summary = {}
    for key in metric_keys:
        values = np.asarray([p[key] for p in per_protein], dtype=np.float64)
        summary['{}_mean'.format(key)] = float(values.mean())
        summary['{}_std'.format(key)] = float(values.std(ddof=ddof)) if len(values) > ddof else 0.0
    return summary


def init_wandb(key_file, entity, project, name, config=None):
    """`wandb.login` + `wandb.init` boilerplate shared by every eval/train script."""
    import wandb
    with open(key_file) as f:
        api_key = f.read().strip()
    wandb.login(key=api_key)
    return wandb.init(entity=entity, project=project, name=name, config=config or {})


def log_per_protein_table(per_protein, metric_keys=METRIC_KEYS, columns=('title', 'length'),
                           table_name='per_protein_metrics'):
    """Upload a wandb.Table of per-protein metrics (one row per protein)."""
    import wandb
    table = wandb.Table(columns=list(columns) + list(metric_keys))
    for p in per_protein:
        table.add_data(*(p[c] for c in columns), *(p[k] for k in metric_keys))
    wandb.log({table_name: table})
