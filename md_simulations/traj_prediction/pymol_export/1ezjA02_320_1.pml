# PyMOL script for 1ezjA02_320_1
# Largest CA RMSD = 23.49 A between frame 10 and frame 66 (0-based).
load /home/cmikulski/Research/Bhaskar_lab/Sheaf_protein_dynamics/MD_sims/pymol_export/1ezjA02_320_1.pdb, traj
dss traj
create frame_10, traj, 11, 1
create frame_66, traj, 67, 1
dss frame_10
dss frame_66
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_10
color firebrick, frame_66
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
