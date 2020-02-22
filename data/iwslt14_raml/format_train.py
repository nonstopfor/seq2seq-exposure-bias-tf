in_file_path = '../iwslt14/samples_iwslt14.txt'
out_file_path = 'train.txt'

out_file = open(out_file_path, 'w', encoding='utf-8')
n_samples = 10


def read_raml_sample_file():
    raml_file = open(in_file_path, 'r', encoding='utf-8')

    train_data = []
    sample_num = -1
    for line in raml_file.readlines():
        line = line[:-1]
        if line.startswith('***'):
            continue
        elif line.endswith('samples'):
            sample_num = eval(line.split()[0])
            assert sample_num == 1 or sample_num == n_samples
        elif line.startswith('source:'):
            train_data.append({'source': line[7:], 'targets': []})
        else:
            train_data[-1]['targets'].append(line.split('|||'))
            if sample_num == 1:
                for i in range(n_samples - 1):
                    train_data[-1]['targets'].append(line.split('|||'))
    return train_data


data = read_raml_sample_file()
for d in data:
    source = d['source'].strip()
    for a in d['targets']:
        target = a[0].strip()
        score = a[1]
        out_file.write(source + '\n' + target + '\n' + str(score).strip() + '\n')
