# PyMOL script for 1ezjA02_320_0
# Largest CA RMSD = 23.36 A between frame 10 and frame 171 (0-based).
load /home/cmikulski/Research/Bhaskar_lab/Sheaf_protein_dynamics/MD_sims/pymol_export/1ezjA02_320_0.pdb, traj
dss traj
create frame_10, traj, 11, 1
create frame_171, traj, 172, 1
dss frame_10
dss frame_171
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_10
color firebrick, frame_171
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
