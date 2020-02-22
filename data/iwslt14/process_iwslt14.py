files=['train','valid','test']
out_files=['train.txt','dev.txt','test.txt']
for i in range(len(files)):
    source_file=open(files[i]+'.de','r',encoding='utf-8')
    target_file=open(files[i]+'.en','r',encoding='utf-8')
    out_file=open(out_files[i],'w',encoding='utf-8')
    source=source_file.readlines()
    target=target_file.readlines()
    assert len(source)==len(target), 'source file and target file must have the same lines'
    for i in range(len(source)):
        a=source[i]
        b=target[i]
        out_file.write(a)
        out_file.write(b)