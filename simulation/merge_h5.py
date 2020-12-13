import h5py
import os

all_rgb_images_exist = []
all_encode_questions_exist = []
all_answers_eixst = []
all_rgb_images_count = []
all_encode_questions_count = []
all_answers_count = []

h5_file_dir = os.path.abspath('../data/vqa')
for group_num in range(7,10):  # one eposide
    for scene_num in range(3,6):     
        h5_file_name = 'group-'+ '0'+str(group_num) + '-scene-'+'0'+ str(scene_num) + '.h5'
        h5_full_file = os.path.join(h5_file_dir , h5_file_name) 
        print(h5_file_name)
        f  = h5py.File(h5_full_file,'r')
        for i in range(20):
            if f['answers'][i] >=4:
                all_rgb_images_exist.append(f['images'][i])
                all_encode_questions_exist.append(f['questions'][i])
                all_answers_eixst.append(f['answers'][i])
            else:
                all_rgb_images_count.append(f['images'][i])
                all_encode_questions_count.append(f['questions'][i])
                all_answers_count.append(f['answers'][i])
        f.close()

f  = h5py.File("global+local_mid_exist.h5",'w')
f['images'] = all_rgb_images_exist
f['questions'] = all_encode_questions_exist
f['answers'] = all_answers_eixst
f.close()

f  = h5py.File("global+local_mid_count.h5",'w')
f['images'] = all_rgb_images_count
f['questions'] = all_encode_questions_count
f['answers'] = all_answers_count
f.close()