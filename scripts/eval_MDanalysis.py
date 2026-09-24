# TODO: 
# 1. Import MDanalysis
# 2. Initialize best sheaf model with weights
# 3. Run OOD
# 4. Compare to pre-trained, fine-tuned DynamicMPNN.

# Get dataset
from  MDAnalysisData import datasets

adk_transitions = datasets.fetch_adk_equilibrium()
