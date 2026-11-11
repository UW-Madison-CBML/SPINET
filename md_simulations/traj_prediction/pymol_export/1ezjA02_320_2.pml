# PyMOL script for 1ezjA02_320_2
# Largest CA RMSD = 23.38 A between frame 1 and frame 402 (0-based).
load /home/cmikulski/Research/Bhaskar_lab/Sheaf_protein_dynamics/MD_sims/pymol_export/1ezjA02_320_2.pdb, traj
dss traj
create frame_1, traj, 2, 1
create frame_402, traj, 403, 1
dss frame_1
dss frame_402
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_1
color firebrick, frame_402
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
