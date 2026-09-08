set -e

ENVNAME=PiFold
export ENVDIR=$ENVNAME

export PATH

mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate


tar -xvf PiFold.tar.gz
cd PiFold/

tar -xvf ../surffold_data.tar.gz

python mask_ipa_pretrain.py \
    --config-name=mask_pretrain \
    comet.use=false \
    dataset.train_dir=./surffold_data/train \
    dataset.val_dir=./surffold_data/validation

IPA_MODEL_PATH=$(find . -name "*.ckpt" -o -name "*.pth" | head -n 1)


python main.py \
    --config-name=diff_config \
    prior_model.path="${IPA_MODEL_PATH}" \
    dataset.train_dir=./surffold_data/train \
    dataset.val_dir=./surffold_data/validation \
    dataset.test_dir=./surffold_data/test
