"""Fetch hashed original PFF train/dev files, never final or bridge contents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import gdown
import requests
from bs4 import BeautifulSoup
from pff_reader import sha


def download(item,dest):
    """Anonymous provider download; normal large-file confirmation when needed."""
    try:gdown.download(id=item['id'],output=str(dest),quiet=True,use_cookies=False)
    except gdown.exceptions.FileURLRetrievalError:
        with requests.Session() as session:
            page=session.get('https://drive.google.com/uc',params={'export':'download','id':item['id']},timeout=30);page.raise_for_status()
            if 'text/html' not in page.headers.get('Content-Type',''):
                data=page.content
                if item['path'].endswith('.jsonl.bz2'):
                    if data[:3]!=b'BZh':raise ValueError('Unexpected tracking payload')
                else:json.loads(data)
                dest.write_bytes(data);return
            if 'Google Drive - Virus scan warning' not in page.text:raise ValueError('Unexpected provider page; no access-gate bypass')
            form=BeautifulSoup(page.text,'html.parser').find('form')
            if not form or form.get('action')!='https://drive.usercontent.google.com/download' or form.get('method')!='get':raise ValueError('Unexpected provider confirmation form')
            params={x['name']:x.get('value','') for x in form.find_all('input') if x.get('name')}
            if set(params)!={'id','export','confirm','uuid'} or params['id']!=item['id']:raise ValueError('Unexpected large-file parameters')
            with session.get(form['action'],params=params,stream=True,timeout=45) as response:
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type',''):raise ValueError('Provider returned another interstitial; stop')
                with dest.open('wb') as stream:
                    for block in response.iter_content(1024*1024):stream.write(block)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',action='store_true',help='Fetch original ten development games instead of36training');a=p.parse_args()
    selected='development' if a.development else 'training';registry=json.loads((ROOT/'tools/pff_source_files.json').read_text());protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())
    allowed={g['game_id'] for g in protocol[selected]};files=[f for f in registry['files'] if f['split']==selected]
    if len(files)!=3*len(allowed) or len({f['path'] for f in files})!=len(files):raise ValueError('Incomplete/duplicate source registry')
    for item in files:
        relative=Path(item['path'])
        if len(relative.parts)!=2 or relative.parts[0] not in ['Tracking Data','Metadata','Rosters'] or relative.name.split('.')[0] not in allowed:raise ValueError('Unapproved path/game; reserve access blocked')
        dest=ROOT/'external/pff'/relative;alt=ROOT/'external/pff/Tracking'/relative.name
        if not dest.exists() and relative.parts[0]=='Tracking Data' and alt.exists():dest=alt
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+'.part')
            if part.exists():raise ValueError('Preserve/inspect interrupted download before restarting')
            download(item,part)
            if part.stat().st_size!=item['bytes'] or sha(part)!=item['sha256']:raise ValueError('Downloaded source differs from the registered original')
            part.rename(dest)
        if dest.stat().st_size!=item['bytes'] or sha(dest)!=item['sha256']:raise ValueError('Local original source hash changed')
        print(selected,item['path'],'verified',flush=True)


if __name__=='__main__':main()
