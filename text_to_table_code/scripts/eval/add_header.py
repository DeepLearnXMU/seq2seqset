import sys

import numpy as np


_, header, hyp, text, oup = sys.argv

team = ['Number of team assists',
            'Percentage of 3 points',
            'Percentage of field goals',
            'Losses',
            'Total points',
            'Points in 1st quarter',
            'Points in 2nd quarter',
            'Points in 3rd quarter',
            'Points in 4th quarter',
            'Rebounds',
            'Turnovers',
            'Wins', ]

player = ['Assists',
              'Blocks',
              'Defensive rebounds',
              '3-pointers attempted',
              '3-pointers made',
              '3-pointer percentage',
              'Field goals attempted',
              'Field goals made',
              'Field goal percentage',
              'Free throws attempted',
              'Free throws made',
              'Free throw percentage',
              'Minutes played',
              'Offensive rebounds',
              'Personal fouls',
              'Points',
              'Total rebounds',
              'Steals',
              'Turnovers', ]

texts = open(text).readlines()
with open(hyp) as f:
    lines = f.readlines()
lines = [x.strip() for x in lines if x.startswith("C-")]
if len(lines) == 0:
    print("No predictions found")
    button = input('Use ground truth headers? (y/n)')
    if button == 'y':
        if header == 'team':
            data_path = 'data/rotowire_team/test.data.header'
        elif header == 'player':
            data_path = 'data/rotowire_player/test.data.header'
        else:
            raise ValueError("Unknown header type")
        data = open(data_path).readlines()
        assert len(data) == len(texts)
        
        
    exit(0)

ids = [int(x.split("C-")[1].split()[0]) for x in lines]
assert sorted(ids) == list(range(len(ids)))
classes = [x.split("\t")[-1] for x in lines]

if header == "team":
    header = team
elif header == "player":
    header = player
else:
    raise ValueError("Unknown header type")
assert len(texts) == len(classes)
with open(oup, 'w') as f:
    for idx, i in enumerate(np.lexsort((np.arange(len(ids)), ids))):
        # preserve order for multi beam
        # can also use `kind='stable'`; but don't wanna do that
        onehot = classes[i]
        out_header = [header[i] for i, x in enumerate(onehot) if x == '1']
        if len(out_header) > 0:
            if texts[idx].strip() == "":
                f.write(f'|  | {" | ".join(out_header)} |\n')
            else:
                f.write(f'|  | {" | ".join(out_header)} | <NEWLINE> {texts[idx]}')
        else:
            f.write(f'|  |\n')
