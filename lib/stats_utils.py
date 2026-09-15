import torch
import numpy as np
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


# ---------------------------------------------------------------------------------------
# The exact validation-metric set scripts/train_residue_classifier.py's `run_val` logs.
#
# Every comparison model in comparisons/ has to report the *same* W&B keys or the runs
# cannot be read side by side in one project. Rather than re-deriving precision/recall/f1
# /top-k/perplexity/confusion-matrix conventions in each model's train script (and drifting
# from the sheaf model's), they all accumulate into this class and log what it returns.
#
# Conventions copied deliberately from `run_val`, not reinvented:
#   * every statistic is accumulated PER PROTEIN -- train_residue_classifier.py runs with
#     batch_size=1, so its "per batch" precision/recall/f1 and its per-protein top-k and
#     perplexity are the same thing. Means and stds are therefore over proteins.
#   * precision/recall/f1 are computed from each protein's own confusion matrix and then
#     averaged (a macro average over proteins), with 0/0 -> 0 via nan_to_num. They are NOT
#     recomputed from the summed matrix.
#   * perplexity is exp() of each protein's mean cross-entropy, averaged over proteins;
#     `epoch_<val_name>_loss` is the plain mean of those cross-entropies.
#   * `std` is torch's default sample std (ddof=1), as in run_val.
# ---------------------------------------------------------------------------------------

def three_letter_codes(one_letter_order):
    """['A', 'C', ...] -> ['ALA', 'CYS', ...].

    The comparison models index their output classes with one-letter alphabets in their own
    orders (DynamicMPNN's `BASE_AMINO_ACIDS`, PiFold's `ALPHABET`, MapDiff's `AMINO_ACIDS`),
    while train_residue_classifier.py names its per-residue W&B keys with the uppercase
    three-letter codes in `ResidueClassifierDataset.AMINO_ACIDS`. Translating here is what
    makes `val_ALA_f1_mean` mean the same thing in every run regardless of class order.
    """
    from Bio.Data import IUPACData
    one_to_three = {v.upper(): k.upper() for k, v in IUPACData.protein_letters_3to1.items()}
    return [one_to_three[aa.upper()] for aa in one_letter_order]


class ResidueMetrics:
    """Accumulator producing run_val's metric dict for a single evaluation pass.

    :param amino_acids: uppercase three-letter residue codes **in the model's own class-index
        order** (see `three_letter_codes`). Doubles as the confusion matrix's axis labels.
    :param val_name: key prefix, e.g. 'val' or 'test'.
    """

    def __init__(self, amino_acids, val_name='val'):
        self.amino_acids = list(amino_acids)
        self.num_classes = len(self.amino_acids)
        self.val_name = val_name

        self.global_confusion_mat = torch.zeros((self.num_classes, self.num_classes))
        self.acc_top_1, self.acc_top_5, self.acc_top_10 = [], [], []
        self.losses = []
        self.f1s = {acid: [] for acid in self.amino_acids}
        self.precisions = {acid: [] for acid in self.amino_acids}
        self.recalls = {acid: [] for acid in self.amino_acids}
        self.pred_seqs, self.pred_pdbs = [], []
        self.scrmsd = []

    def __len__(self):
        return len(self.losses)

    def add_protein(self, logits, targets, pdb_id=None, nll=None, sequence=None):
        """Score one protein.

        :param logits: (R, num_classes) scores for the residues being predicted. Log-probs
            work equally well for top-k and the confusion matrix; pass `nll` yourself in that
            case so the perplexity is not taken as if these were raw logits.
        :param targets: (R,) ground-truth class indices.
        :param pdb_id: protein id (``<pdb>_<chain>`` for ATLAS, the CATH domain for mdCATH).
            Present in the `pred_seqs` table alongside the design.
        :param nll: per-residue mean negative log-likelihood; defaults to the cross-entropy
            of `logits`.
        :param sequence: the full one-letter design. Defaults to the argmax of `logits`,
            which is what every comparison model wants -- unlike the sheaf classifier, none of
            them mask part of the sequence and splice the ground truth back in.
        """
        logits = logits.detach().cpu().float()
        targets = targets.detach().cpu().long()
        if logits.shape[0] == 0:
            return

        if nll is None:
            nll = F.cross_entropy(logits, targets).item()
        self.losses.append(float(nll))

        self.acc_top_1.append(top_k_acc(logits, targets, 1))
        self.acc_top_5.append(top_k_acc(logits, targets, 5))
        self.acc_top_10.append(top_k_acc(logits, targets, 10))

        preds = logits.argmax(dim=-1)
        confusion_mat = get_confusion_matrix(targets, preds, self.num_classes)
        self.global_confusion_mat += confusion_mat

        diag = confusion_mat.diag()
        recall = torch.nan_to_num(diag / confusion_mat.sum(dim=1), 0.0)
        precision = torch.nan_to_num(diag / confusion_mat.sum(dim=0), 0.0)
        f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
        for k, amino_acid in enumerate(self.amino_acids):
            self.precisions[amino_acid].append(precision[k].item())
            self.recalls[amino_acid].append(recall[k].item())
            self.f1s[amino_acid].append(f1[k].item())

        if pdb_id is not None:
            if sequence is None:
                from Bio.SeqUtils import seq1
                sequence = ''.join(seq1(self.amino_acids[i]) for i in preds.tolist())
            self.pred_pdbs.append(pdb_id)
            self.pred_seqs.append(sequence)

    def add_scrmsd(self, values):
        """Self-consistency RMSDs from `lib.scrmsd.evaluate_scrmsd`, in any batching."""
        self.scrmsd.extend(float(v) for v in values)

    def to_log_dict(self, cm_title=None, epoch=None):
        """run_val's `prf_dict`, ready to hand straight to `run.log`."""
        import pandas as pd
        import wandb
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        val_name = self.val_name
        prf_dict = {}
        precisions = {key: torch.tensor(value) for key, value in self.precisions.items()}
        recalls = {key: torch.tensor(value) for key, value in self.recalls.items()}
        f1s = {key: torch.tensor(value) for key, value in self.f1s.items()}
        for amino_acid in self.amino_acids:
            prf_dict['{}_{}_f1_mean'.format(val_name, amino_acid)] = f1s[amino_acid].mean().item()
            prf_dict['{}_{}_precision_mean'.format(val_name, amino_acid)] = precisions[amino_acid].mean().item()
            prf_dict['{}_{}_recall_mean'.format(val_name, amino_acid)] = recalls[amino_acid].mean().item()
            prf_dict['{}_{}_f1_std'.format(val_name, amino_acid)] = f1s[amino_acid].std().item()
            prf_dict['{}_{}_precision_std'.format(val_name, amino_acid)] = precisions[amino_acid].std().item()
            prf_dict['{}_{}_recall_std'.format(val_name, amino_acid)] = recalls[amino_acid].std().item()

        if self.pred_pdbs:
            pred_string_df = pd.DataFrame({"pdb": self.pred_pdbs, "seq": self.pred_seqs})
            # run_val logs the validation designs under the bare key `pred_seqs`; keep that
            # name for the val pass so the tables line up, and namespace any other pass.
            table_key = 'pred_seqs' if val_name == 'val' else '{}_pred_seqs'.format(val_name)
            prf_dict[table_key] = wandb.Table(dataframe=pred_string_df)

        perplexities = torch.exp(torch.tensor(self.losses))
        prf_dict['{}_perp_mean'.format(val_name)] = perplexities.mean().item()
        prf_dict['{}_perp_std'.format(val_name)] = perplexities.std().item()

        if self.scrmsd:
            scrmsd = torch.tensor(self.scrmsd)
            prf_dict['{}_rmsd_mean'.format(val_name)] = scrmsd.mean().item()
            prf_dict['{}_rmsd_std'.format(val_name)] = scrmsd.std().item()

        acc_top_1 = torch.tensor(self.acc_top_1)
        acc_top_5 = torch.tensor(self.acc_top_5)
        acc_top_10 = torch.tensor(self.acc_top_10)
        prf_dict['{}_top1_acc_mean'.format(val_name)] = acc_top_1.mean().item()
        prf_dict['{}_top5_acc_mean'.format(val_name)] = acc_top_5.mean().item()
        prf_dict['{}_top10_acc_mean'.format(val_name)] = acc_top_10.mean().item()
        prf_dict['{}_top1_acc_std'.format(val_name)] = acc_top_1.std().item()
        prf_dict['{}_top5_acc_std'.format(val_name)] = acc_top_5.std().item()
        prf_dict['{}_top10_acc_std'.format(val_name)] = acc_top_10.std().item()

        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self.global_confusion_mat.numpy().astype(int),
            display_labels=self.amino_acids,
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.title(cm_title or 'Amino Acid Confusion Matrix')
        prf_dict['{}_aa_confusion_matrix'.format(val_name)] = wandb.Image(fig)
        plt.close(fig)

        prf_dict['epoch_{}_loss'.format(val_name)] = (sum(self.losses) / len(self.losses)) if self.losses else 0
        if epoch is not None:
            prf_dict['epoch'] = epoch
        return prf_dict

    def summary_line(self, num_params):
        """run_val's LaTeX table row: params & top-1 & top-5 & top-10 & perplexity."""
        perplexities = torch.exp(torch.tensor(self.losses))
        a1 = torch.tensor(self.acc_top_1)
        a5 = torch.tensor(self.acc_top_5)
        a10 = torch.tensor(self.acc_top_10)
        line = ('{} & ${:.3f} \\pm {:.3f}$ & ${:.3f} \\pm {:.3f}$ & ${:.3f} \\pm {:.3f}$ '
                '& ${:.3f} \\pm {:.3f}$').format(
            num_params, a1.mean().item(), a1.std().item(), a5.mean().item(), a5.std().item(),
            a10.mean().item(), a10.std().item(), perplexities.mean().item(), perplexities.std().item())
        if self.scrmsd:
            scrmsd = torch.tensor(self.scrmsd)
            line += ' & ${:.3f} \\pm {:.3f}$'.format(scrmsd.mean().item(), scrmsd.std().item())
        return line
