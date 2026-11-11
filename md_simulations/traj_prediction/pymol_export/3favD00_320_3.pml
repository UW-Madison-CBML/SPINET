# PyMOL script for 3favD00_320_3
# Largest CA RMSD = 23.44 A between frame 111 and frame 487 (0-based).
load /home/cmikulski/Research/Bhaskar_lab/Sheaf_protein_dynamics/MD_sims/pymol_export/3favD00_320_3.pdb, traj
dss traj
create frame_111, traj, 112, 1
create frame_487, traj, 488, 1
dss frame_111
dss frame_487
hide everything
show cartoon
set cartoon_trace_atoms, 0
color grey70, traj
color marine, frame_111
color firebrick, frame_487
disable traj
orient
# Type "enable traj" and press play to see the whole trajectory.
# PyMOL state k+1 is trajectory frame k.
