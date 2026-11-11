# PyMOL script for 3j7yg00_320_4
# Largest CA RMSD = 25.02 A between frame 59 and frame 485 (0-based).
load /home/cmikulski/Research/Bhaskar_lab/Sheaf_protein_dynamics/MD_sims/pymol_export/3j7yg00_320_4.pdb, traj
dss traj
create frame_59, traj, 60, 1
create frame_485, traj, 486, 1
dss frame_59
dss frame_485
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_59
color firebrick, frame_485
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
