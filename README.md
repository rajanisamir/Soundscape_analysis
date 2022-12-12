# Soundscape analysis

## Generating images of spectrograms from an audio file

```
from audio_utilities import build_spectrogram_images_from_audio_file
inputpath = '/some/path/to/the/input/audio/file.wav'
outputpath = '/some/path/to/the/output/folder'
window_t = 10
min_f = 0
max_f = 16000
build_spectrogram_images_from_audio_file(inputpath,outputpath,10,0,16000)
```

## Generating images of spectrograms from files in directory tree

We can run it from a jupyter-notebook on one CPU by means of the following commands

```
Arguments = namedtuple('Arguments', ['input_directory',
                                     'output_directory',
                                     'window_t',
                                     'min_f',
                                     'max_f'
                                    ])

args = Arguments(input_directory = '/media/dario/T7_Touch/Test_folder/',
                 output_directory = '/media/dario/T7_Touch/Test_folder1/',
                 window_t = 10,
                 min_f = 0.0,
                 max_f = 20000.0
                )

build_spectrogram_images_from_directory_tree.main(args)
```
or from the command line using `mpi4py` by means of the following command on many CPUs

`mpiexec -n number of ranks python build_spectrogram_images_from_directory_tree.py --input_directory /the/input/directory/ --output_directory /the/output/directory/ --window_t temporal window of the spectrograms --min_f minimum frequency --max_f maximum frequency`

## Joint Embedding Training on shared memory

The following command runs a training instance on a shared memory system on a single GPU.
This is generally used in a local machine

`python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --data_path /path/to/your/images/ --output_dir /path/to/your/output/models/ --use_fp16 false --batch_size_per_gpu number of images per batch`

For a HPC cluster, on a node with shared memory, on a single GPU, we use the following command on ThetaGPU (a compute cluster at Argonne National Laboratory)

This command submit the job to run during 10 minutes.

`qsub -n 1 -q single-gpu -t 10 -A MyProjectName ./my_script_to_run.sh`

where `my_script_to_run.sh` is

```
  1 #!/bin/sh
  2 
  3 # Common paths
  4 spectrogram_images_path='/path/to/spectrogram/images/folder'
  5 singularity_image_path='/path/to/singularity/image.sif'
  6 dino_path='/path/to/dino'
  7 train_dino_path='/path/to/main_dino.py'
  8 model_path='/path/to/where/the/models/checkpoints/will/be/saved'
  9 
 10 cd $dino_path
 11 singularity exec --nv -B $spectrogram_images_path:/Spectrogram_Images,$model_path:/Model $singularity_image_path python -m torch.distributed.launch --nproc_per_node=1 $train_dino_path --arch vit_small --data_path /Spectrogram_Images --output_dir /Model --use_fp16 false --epo    chs 15 --saveckp_freq 1 --warmup_epochs 5 --batch_size_per_gpu 64 --local_crops_number 20 --lr 0.00005
 12 
 13 
 14 
 
```
