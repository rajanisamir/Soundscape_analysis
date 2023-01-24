import os
from pathlib import Path
from shutil import copy

def generate_labelled_dataset(label_predictions, file_paths, spectrograms_base_path, output_dir, verbose=False):
    for label, fpath in zip(label_predictions, file_paths):
        file_path=''.join([chr(int(x)) for x in fpath]).replace('~','')
        file_path=os.path.join(spectrograms_base_path,file_path)
        output_path=os.path.join(output_dir,'Class_'+str(label))
        Path(output_path).mkdir(parents=True, exist_ok=True)
        copy(file_path,output_path)
        if verbose:
            print('Label is ', label)
            print('File path is ', file_path)
            print('To output directory in ', output_path)
