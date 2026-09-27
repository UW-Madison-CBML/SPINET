## `SPINET`
Sheaf neural network for protein inverse folding using molecular dynamics

## Dependencies
All core model code can be ran via the Docker image `jenslundsgaard/sheaf_training:v3.2`.
## Using the model
Initialize the model with the proper args:
```
from invariant_features_sheaf_model import NodeSheafClassifier
model = NodeSheafClassifier(**args)
```
Pull the model weights from HuggingFace
```
weights_path = hf_hub_download("JensLundsgaard/our_model_mdcath", "pytorch_model.bin", local_dir=os.path.abspath("./"))
model.load_state_dict(torch.load(weights_path, weights_only=True))
```

