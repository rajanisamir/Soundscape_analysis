import os
import numpy as np
from pathlib import Path
from shutil import copy

def generate_labelled_dataset(label_predictions, file_paths, spectrograms_base_path, output_dir, checkpoint_freq=1000, verbose=False):
    if os.path.isfile(os.path.join(output_dir, 'start_label_predictions.npy')):
        start_label_predictions=np.load(os.path.join(output_dir, 'start_label_predictions.npy'))
        print('Checkpoint at ', start_label_predictions.item())
    else:
        start_label_predictions=np.array(0)
        print('No checkpoint. Starting from 0.')


    print('Checkpointing at every {} samples' .format(checkpoint_freq))
    for i, (label, fpath) in enumerate(zip(label_predictions, file_paths)):
        if i >= start_label_predictions.item():
            try:
                file_path=''.join([chr(int(x)) for x in fpath]).replace('~','')
                FILE_PATH=os.path.join(spectrograms_base_path,file_path)
                auxiliary, _ = os.path.split(file_path)
                output_path=os.path.join(output_dir,'Class_'+str(label),auxiliary)
                Path(output_path).mkdir(parents=True, exist_ok=True)
                copy(FILE_PATH,output_path)
                if verbose:
                    print('Label is ', label)
                    print('File path is ', file_path)
                    print('To output directory in ', output_path)
            except:
                print('Something went wrong with the following data path') 
                print('Label is ', label)
                print('File path is ', file_path)
                print('To output directory in ', output_path)

            if i%checkpoint_freq==0:
                np.save(os.path.join(output_dir, 'start_label_predictions.npy'), np.array(i))

    print('DONE!')

