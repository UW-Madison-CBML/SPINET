import requests 
import pandas as pd
import threading
import os
def download_file(url):
    local_filename = os.path.join("md_data",f"{url.split('/')[-1]}.zip")
    with requests.get(url, stream=True) as r: # stream=True is very important
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename


def main():
    threads = {}
    base_url = "https://www.dsimb.inserm.fr/ATLAS/api" 
    atlas_df = pd.read_csv(os.path.abspath("atlas.csv"))
    md_urls = [base_url + f"/ATLAS/analysis/{pdb}" for pdb in atlas_df["pdb"].to_list()]

    for i in md_urls:
        print(i)
        threads[i] = threading.Thread(target=download_file, args=(i,))
    for i in threads.keys():
        threads[i].start()
    for i in threads.keys():
        threads[i].join()
    
        

if __name__ == "__main__":
    main()
