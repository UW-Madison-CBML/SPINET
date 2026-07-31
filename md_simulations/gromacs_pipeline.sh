#!/bin/bash

PROTEIN=$1

# --- Configuration ---
PDB_INPUT="$PROTEIN.pdb"
FF="amber99sb-ildn"
WATER="tip3p"
OUT_DIR="md_data/$PROTEIN"
MDP_DIR="./mdp"
BOX=dodecahedron   # changes amount of water, see description for other options
BOX_DIST=1.0       # box edge distance in nm
CONC=0.1           # molar NaCl concentration

# Create output directory if it doesn't exist
mkdir -p $OUT_DIR


gmx --version | grep -i "GPU support"

echo "Step 1: Generating Topology..."
gmx pdb2gmx -f $PDB_INPUT -o $OUT_DIR/processed.gro -p $OUT_DIR/topol.top -ff amber99sb-ildn -water none

echo "Step 2: Defining Box..."
gmx editconf -f $OUT_DIR/processed.gro -o $OUT_DIR/boxed.gro -bt cubic -d 2.0

echo "Step 3: Minimization..."
gmx grompp -f $MDP_DIR/minim.mdp -c $OUT_DIR/boxed.gro -p $OUT_DIR/topol.top -o $OUT_DIR/em.tpr
gmx mdrun -v -deffnm $OUT_DIR/em -nb cpu -pme cpu

echo "Step 4: Production Run..."
gmx grompp -f $MDP_DIR/md.mdp -c $OUT_DIR/em.gro -p $OUT_DIR/topol.top -o $OUT_DIR/production.tpr
gmx mdrun -v -deffnm $OUT_DIR/production -nb gpu -pme gpu -bonded gpu -update gpu

### 1. Structure Conversion
#echo "******************************************"
#echo "Step 1: Generating Topology..."
#gmx pdb2gmx -f $PDB_INPUT -o $OUT_DIR/processed.gro -p $OUT_DIR/topol.top -ff $FF -water $WATER
#mv posre.itp $OUT_DIR/
#
## 2. Creating the Box & Solvating
#echo "******************************************"
#echo "Step 2: Defining Box and Adding Solvent..."
#gmx editconf -f $OUT_DIR/processed.gro -o $OUT_DIR/boxed.gro -bt $BOX -d $BOX_DIST
#gmx solvate -cp $OUT_DIR/boxed.gro -cs spc216.gro -o $OUT_DIR/solvated.gro -p $OUT_DIR/topol.top
#
## 3. Adding Ions
#echo "******************************************"
#echo "Step 3: Adding Ions..."
#gmx grompp -f $MDP_DIR/ions.mdp -c $OUT_DIR/solvated.gro -p $OUT_DIR/topol.top -o $OUT_DIR/ions.tpr -po $OUT_DIR/ions_out.mdp
## Replaces SOL with ions to neutralize
#echo "SOL" | gmx genion -s $OUT_DIR/ions.tpr -o $OUT_DIR/ionized.gro -p $OUT_DIR/topol.top -pname NA -nname CL -neutral -conc $CONC
#nvidia-smi
## 4. Energy Minimization
#echo "******************************************"
#echo "Step 4: Energy Minimization..."
#gmx grompp -f $MDP_DIR/minim.mdp -c $OUT_DIR/ionized.gro -p $OUT_DIR/topol.top -o $OUT_DIR/em.tpr -po $OUT_DIR/em_out.mdp
#gmx mdrun -nb gpu -pme gpu -v -deffnm $OUT_DIR/em
#grep "Steepest Descents converged to Fmax" $OUT_DIR/em.log
#nvidia-smi
## 5. NVT Equilibration
#echo "******************************************"
#echo "Step 5: NVT Equilibration..."
#gmx grompp -f $MDP_DIR/nvt.mdp -c $OUT_DIR/em.gro -r $OUT_DIR/em.gro -p $OUT_DIR/topol.top -o $OUT_DIR/nvt.tpr -po $OUT_DIR/nvt_out.mdp
#gmx mdrun -nb gpu -pme gpu -v -deffnm $OUT_DIR/nvt
#
## 6. NPT Equilibration
#echo "******************************************"
#echo "Step 6: NPT Equilibration..."
#gmx grompp -f $MDP_DIR/npt.mdp -c $OUT_DIR/nvt.gro -r $OUT_DIR/nvt.gro -t $OUT_DIR/nvt.cpt -p $OUT_DIR/topol.top -o $OUT_DIR/npt.tpr -po $OUT_DIR/npt_out.mdp
#gmx mdrun -nb gpu -pme gpu -v -deffnm $OUT_DIR/npt
#
## 7. Production MD
#echo "******************************************"
#echo "Step 7: Production Run..."
#gmx grompp -f $MDP_DIR/md.mdp -c $OUT_DIR/npt.gro -t $OUT_DIR/npt.cpt -p $OUT_DIR/topol.top -o $OUT_DIR/production.tpr -po $OUT_DIR/md_out.mdp
#gmx mdrun -nb gpu -pme gpu -v -deffnm $OUT_DIR/production


# =====================================================================
# Step 8: Data Extraction for Conformation & Energy Landscape Analysis
# =====================================================================
echo "******************************************"
echo "Step 8: Extracting Landscape Data..."


# 1. Grab Energy Descent from Energy Minimization (Potential Energy vs. Step)
# '11' is typically Potential Energy in minim.edr. 
# We echo '11' and an empty newline to select it automatically.
echo -e "Potential\n" | gmx energy -f $OUT_DIR/em.edr -o $OUT_DIR/em_potential.xvg

# 2. Grab Thermodynamic Trajectory (Temperature, Pressure, Total Energy)
# These track how the system settles during equilibration and production.
echo -e "Temperature\nPressure\nPotential\nKinetic-En.\nTotal-Energy\n" | \
gmx energy -f $OUT_DIR/production.edr -o $OUT_DIR/production_thermo.xvg

# 3. Clean the Production Trajectory (Fix Periodic Boundary Conditions)
# Centering the protein ensures structural analysis calculations are correct.
# Select '1' (Protein) for centering and '0' (System) for output.
echo -e "Protein\nSystem\n" | gmx trjconv \
    -s $OUT_DIR/production.tpr \
    -f $OUT_DIR/production.xtc \
    -o $OUT_DIR/fixed_production.xtc \
    -pbc mol -center

# RMSD: Structural distance from starting structure over time (Select 1 for Protein)
echo -e "Protein\nProtein\n" | gmx rms \
    -s $OUT_DIR/production.tpr \
    -f $OUT_DIR/fixed_production.xtc \
    -o $OUT_DIR/rmsd.xvg

# Radius of Gyration: Compactness/folding state over time (Select 1 for Protein)
echo -e "Protein\n" | gmx gyrate \
    -s $OUT_DIR/production.tpr \
    -f $OUT_DIR/fixed_production.xtc \
    -o $OUT_DIR/gyrate.xvg

# RMSF: Per-residue flexibility profile (Select 1 for Protein)
echo -e "Protein\n" | gmx rmsf \
    -s $OUT_DIR/production.tpr \
    -f $OUT_DIR/fixed_production.xtc \
    -o $OUT_DIR/rmsf.xvg -res

echo "Data extraction complete. Files saved in $OUT_DIR"

rm -f $OUT_DIR/#*#              # all GROMACS backup files
rm -f $OUT_DIR/*.gro            # processed, boxed, solvated, ionized, em.gro, nvt.gro, npt.gro
rm -f $OUT_DIR/em.tpr $OUT_DIR/nvt.tpr $OUT_DIR/npt.tpr $OUT_DIR/ions.tpr
rm -f $OUT_DIR/*.trr            # uncompressed full-precision trajectories from em/nvt/npt if generated
rm -f $OUT_DIR/*_out.mdp        # ions_out.mdp, em_out.mdp, nvt_out.mdp, npt_out.mdp — redundant once you've kept md_out.mdp
rm -f $OUT_DIR/mdout.mdp
rm -f $OUT_DIR/posre.itp

