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
    print("logits: ", logits.shape, ", targets", targets.shape, ", k", k)
    if (k == 1):
        preds = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[1]).float()
        return torch.einsum("bi,bi->b", preds, F.one_hot(targets, num_classes=logits.shape[1]).float()).sum().item() / logits.shape[0]

    hot_logits = torch.zeros_like(logits) 
    indices = torch.topk(logits, k, dim=-1).indices
    hot_logits = hot_logits.scatter_(1, indices, 1).float()
    correct_mask = torch.einsum("bi,bi->b", hot_logits, F.one_hot(targets, num_classes=logits.shape[1]).float())
    return correct_mask.sum().item() / correct_mask.shape[0] 
    


