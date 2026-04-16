#python3 -c "
import json, os

uid  = '00792fa8-988c-4c85-8e80-73eb3ac53e80'
path = '/DATA/G17/Group17_Who_is_Talking_To_Me/starter_code/data/json_original'

# how many tracks per uid?
track_dir = os.path.join(path, uid)
tracks    = os.listdir(track_dir)
print(f'Tracks for {uid}: {len(tracks)}')

# check one track
data = json.load(open(os.path.join(track_dir, tracks[0])))
print(f'Track {tracks[0]}: {len(data)} frames')
print(f'First entry: {data[0]}')
print(f'Last entry:  {data[-1]}')

# check if Person ID varies across tracks
pids = set()
for t in tracks[:10]:
    d = json.load(open(os.path.join(track_dir, t)))
    if d:
        pids.add(d[0].get('Person ID', 'unknown'))
print(f'Person IDs in first 10 tracks: {pids}')
