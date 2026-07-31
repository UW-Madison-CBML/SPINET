import sys
sys.path.append("../lib")

from pdb_api import retrieve_pdb_file
if __name__ == "__main__": 
    assert len(sys.argv) >= 2, "must give at least one protein"
    sub_file = "md_submit.sub"
    out_string = "" 
    out_string += f"JOB job_pdb grab_pdbs.sub\n"  
    out_string += f"VARS job_pdb proteins=\"{' '.join(sys.argv[1:])}\"\n"  
    for job_id, job in enumerate(sys.argv[1:]):
        out_string += f"JOB job{job_id} {sub_file}\n"  
        out_string += f"VARS job{job_id} protein=\"{job}\"\n"  
        #retrieve_pdb_file(job, file_format="pdb", parent_dir="inputs")
    
    out_string += f"PARENT job_pdb CHILD job0\n"
    for job_id in range(len(sys.argv)-2):
        out_string += f"PARENT job{job_id} CHILD job{job_id + 1}\n"
    with open("submit.dag", "w", encoding="utf-8") as file:
        file.write(out_string)
