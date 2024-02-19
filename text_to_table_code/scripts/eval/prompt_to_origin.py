import sys

src_file = open(sys.argv[1], 'r').readlines()
hyp_file = open(sys.argv[2], 'r').readlines()

assert len(src_file) == len(hyp_file)
new_hyp_file = []
for i in range(len(src_file)):
    src = src_file[i].strip()
    hyp = hyp_file[i].strip()
    keys = src.split(' <NEWLINE> ')[1].split(' | ')
    values = hyp.split('|')
    # assert len(keys) == len(values), print(hyp, keys, values)
    key_value = []
    if len(keys) > len(values):
        values += [''] * (len(keys) - len(values))
    for i in range(len(keys)):
        key_value.append(keys[i] + ' ' + values[i])
    key_value = [f'| {keys[i]} | {values[i].strip()} |' for i in range(len(keys))]
    key_value = ' <NEWLINE> '.join(key_value)
    new_hyp_file.append(key_value)

open(sys.argv[2], 'w').write('\n'.join(new_hyp_file))
