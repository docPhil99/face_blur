import json
import os
from pathlib import Path
import datetime
import shutil

org_dir = Path('/media/phil/IntentMAPS/intentMAPS/anon_run1/real_proc1/')
output_dir = Path('/media/phil/IntentMAPS/intentMAPS/anon_run1/real_proc1_corrected/')
for path in org_dir.iterdir():
    if not path.is_dir():
        continue
    print(path)
    json_path = path / Path('timestamps.json')
    try:
        with open(json_path,'rt') as js:
            timestamps = json.load(js)
    except:
        print(f'{json_path} not good')
        continue
    timestamp = timestamps['000000.png']
    date_dir = output_dir/Path(datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%d_%m_%Y'))
    date_dir.mkdir(parents=True,exist_ok=True)
    date_dir = date_dir/path.stem
    print(date_dir)
    os.rename(path,date_dir)
    #break



