import re
import os

#training set
dir = 'F:\DeepLearningDataset\ICVL\old\Training'
txt_dir = 'F:\DeepLearningDataset\ICVL\old\Training\labels.txt'

subject_names=['201403121135','201403121140','201405151126','201406030937',
               '201406031456','201406031503','201406031747','201406041509',
               '201406181554','201406181600','201406191014','201406191044']
data=[]
with open(txt_dir,'r') as f:
    lines=f.readlines()
#get raw joint(without orientation) data
for line in lines:
    sline=line.split()
    name=sline[0].split('/')
    if len(name)==2:
        data.append(sline)

for i_subject in range(len(subject_names)):
    save_file_name=subject_names[i_subject]+'.txt'
    save_data_dir=os.path.join(dir,save_file_name)
    with open(save_data_dir,'w') as fw:
        subject_data=[]
        for line_data in data:
            name=re.split('[._/]',line_data[0])
            if name[0]==subject_names[i_subject]:
                subject_data.append(line_data)
        for s_data in subject_data:
            index=re.split('[._/]',s_data[0])
            fw.write(index[0])
            fw.write(' ')
            fw.write(index[2])
            fw.write(' ')
            for ss_data in s_data[1:]:
                fw.write(ss_data)
                fw.write(' ')
            fw.write('\n')
    print(subject_names[i_subject])


#testing set
dir = 'F:\DeepLearningDataset\ICVL\old\Testing'
txt_dir1 = 'F:\DeepLearningDataset\ICVL\old\Testing\\test_seq_1.txt'
#txt_dir2 = 'F:\DeepLearningDataset\ICVL\old\Testing\\test_seq_2.txt'

data=[]
with open(txt_dir1,'r') as f:
    lines=f.readlines()
save_file_name = '1.txt'
save_data_dir = os.path.join(dir, save_file_name)
valid_lines=[]
for line in lines:
    if len(line)==1:
        continue
    else:
        sline = line.split()
        name=re.split('[._/]',line[0])
        valid_lines.append(sline)

with open(save_data_dir,'w') as fw:
    subject_data = []
    for vline in valid_lines:
        index = re.split('[._/]', vline[0])
        fw.write(index[2])
        fw.write(' ')
        fw.write(index[4])
        fw.write(' ')
        for ss_data in vline[1:]:
            fw.write(ss_data)
            fw.write(' ')
        fw.write('\n')
