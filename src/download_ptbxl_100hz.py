import os
import sys
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

db_path = 'ptbxl_database.csv'
df = pd.read_csv(db_path)

output_dir = '.'
base_url = 'https://physionet.org/content/ptb-xl/1.0.3/'

files_to_download = []
for idx, row in df.iterrows():
    rel_path = str(row['filename_lr']).replace('\\', '/')
    for ext in ['.dat', '.hea']:
        local_file = os.path.normpath(rel_path + ext)
        if not os.path.exists(local_file):
            remote_url = base_url + rel_path + ext
            files_to_download.append((remote_url, local_file))

print(f'Total files needed: {len(df)*2}, Files to download: {len(files_to_download)}')

def download_file(item):
    remote_url, local_file = item
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    try:
        req = urllib.request.Request(remote_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as res, open(local_file, 'wb') as f:
            f.write(res.read())
        return True
    except Exception as e:
        return (remote_url, str(e))

if files_to_download:
    print(f'Starting multi-threaded download of {len(files_to_download)} files...')
    completed, failed = 0, 0
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(download_file, item): item for item in files_to_download}
        for future in as_completed(futures):
            res = future.result()
            if res is True:
                completed += 1
            else:
                failed += 1
            if (completed + failed) % 2000 == 0:
                print(f'Progress: {completed + failed}/{len(files_to_download)}')
    print(f'Download finished. Completed: {completed}, Failed: {failed}')
else:
    print('All files already present locally.')
