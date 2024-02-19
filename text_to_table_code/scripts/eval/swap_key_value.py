import sys

def swap_key_value(tgt): # 先生成value再生成key
    tgt = [i for i in tgt.split(' <NEWLINE> ')]
    new_tgt = []
    for i in tgt:
        ii = i.split('|')
        if len(ii) >= 3:
            ii[1], ii[2] = ii[2], ii[1]
            new_tgt.append('|'.join(ii))
        else:
            new_tgt.append(i)
    return ' <NEWLINE> '.join(new_tgt)

filename = sys.argv[1]
with open(filename) as f:
    line_list = f.read().splitlines()
    line = [swap_key_value(i) for i in line_list]
    with open(filename, 'w') as f:
        f.write('\n'.join(line))
