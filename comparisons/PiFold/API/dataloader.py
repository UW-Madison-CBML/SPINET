"""Overlaid onto a pristine PiFold checkout (see ../run_pifold.sh).

Adds the ATLAS and MDCATH datasets, both served by `API.relaxed_dataset.RelaxedStructures`
-- from relaxed (deposited) PDB structures by default, or from one random MD trajectory frame
per protein with `--structure_source frame`. They are trained and evaluated **separately** --
one run each, `--data_name ATLAS` or `--data_name MDCATH` -- matching
scripts/train_residue_classifier.py's `--ds-name`. Each has three disjoint splits (see
lib/dataset_splits.py): 'test' is held out from training and from model selection.
"""
import copy
import os.path as osp

from .cath_dataset import CATH
from .ts_dataset import TS
from .relaxed_dataset import RelaxedStructures

from .dataloader_gtrans import DataLoader_GTrans
from .featurizer import featurize_GTrans

# --data_name -> lib.dataset_splits dataset name
RELAXED_DATASETS = {'ATLAS': 'atlas', 'MDCATH': 'mdcath'}


def load_data(data_name, method, batch_size, data_root, num_workers=8, **kwargs):
    if data_name == 'CATH' or data_name == 'TS':
        cath_set = CATH(osp.join(data_root, 'cath'), mode='train', test_name='All')
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [cath_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        if data_name == 'TS':
            test_set = TS(osp.join(data_root, 'ts'))
        collate_fn = featurize_GTrans
    elif data_name in RELAXED_DATASETS:
        relaxed_set = RelaxedStructures(
            data_root, mode='train',
            max_length=kwargs.get('max_length', 500),
            ds_name=RELAXED_DATASETS[data_name],
            index_csv=kwargs.get('index_csv') or None,
            traj_h5=kwargs.get('traj_h5') if kwargs.get('traj_h5') is not None else None,
            pdb_cache=kwargs.get('pdb_cache') or None,
            val_fold=kwargs.get('val_fold', 0),
            test_fold=kwargs.get('test_fold', 4),
            structure_source=kwargs.get('structure_source', 'relaxed'),
            frame_seed=kwargs.get('frame_seed', 42),
            frame_index=kwargs.get('frame_index'),
        )
        train_set, valid_set, test_set = map(lambda x: copy.copy(x), [relaxed_set] * 3)
        valid_set.change_mode('valid')
        test_set.change_mode('test')
        collate_fn = featurize_GTrans
    else:
        raise ValueError('unknown dataset {}'.format(data_name))

    train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


def make_cath_loader(test_set, method, batch_size, max_nodes=3000, num_workers=8):
    collate_fn = featurize_GTrans
    test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return test_loader
