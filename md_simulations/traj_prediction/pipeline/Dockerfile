# Image for the prep, MD and analysis jobs (not for ColabFold, which uses its own image).
#
#   docker build -t cmikulski4/traj-md:v1 .
#   docker push cmikulski4/traj-md:v1
#
# HTMD (system building, ships the CHARMM22* files) comes from the acellera channel
# and is free for non-commercial use. OpenMM comes from conda-forge with CUDA 12.
FROM mambaorg/micromamba:1.5.8

RUN micromamba install -y -n base -c acellera -c conda-forge \
        "python>=3.10" htmd openmm "cuda-version=12.*" \
        mdtraj scipy scikit-learn pandas h5py matplotlib-base && \
    micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH
# Fail the build if a package the pipeline needs does not import.
RUN python -c "import openmm, mdtraj, h5py, sklearn, pandas, matplotlib; \
from htmd.builder import charmm; from htmd.builder.solvate import solvate; \
from moleculekit.tools.preparation import systemPrepare; \
import os; home = charmm.htmdCharmmHome(); \
assert os.path.isfile(os.path.join(home, 'top/top_all22star_prot.rtf')); \
assert os.path.isfile(os.path.join(home, 'par/par_all22star_prot.prm')); \
print('openmm', openmm.__version__)"
