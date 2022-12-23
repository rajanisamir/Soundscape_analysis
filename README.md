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

## Generating images of spectrograms from audio files in directory tree

We can run it from a jupyter-notebook on one CPU by means of the following commands

```
Arguments = namedtuple('Arguments', ['input_directory',
                                     'output_directory',
                                     'window_t',
                                     'min_f',
                                     'max_f'
                                    ])

args = Arguments(input_directory = '/the/input/directory/',
                 output_directory = '/the/output/directory/',
                 window_t = 10,
                 min_f = 0.0,
                 max_f = 20000.0
                )

build_spectrogram_images_from_directory_tree.main(args)
```
or from the command line using `mpi4py` by means of the following command on many CPUs

`mpiexec -n number of ranks python build_spectrogram_images_from_directory_tree.py --input_directory /the/input/directory/ --output_directory /the/output/directory/ --window_t temporal window of the spectrograms --min_f minimum frequency --max_f maximum frequency`

or on a node in a cluster using 4 CPUs, you first submit your job in a queue

`qsub -n 1 -q single-gpu -t 10 -A MyProjectName ./my_script_to_run.sh`

where `my_script_to_run.sh` is


```
#!/bin/sh
 
# Common paths
singularity_image_path='/path/to/singularity/image.sif'
sound_analysis_path='/sound/analysius/path'
spectrogram_builder_script_path='/path/to/Soundscape_analysis/build_spectrogram_images_from_directory_tree.py'
input_directory='/input/directory/fo/the/audio/files'
output_path='/output/directory/for/the/spectrogram/images'

cd $sound_analysis_path
singularity exec --nv -B $input_directory:/Audio_files,$output_path:/Outputs $singularity_image_path mpiexec -n 4 python $spectrogram_builder_script_path --input_directory /Audio_files --output_directory /Outputs --window_t time window --min_f minimum frequency --max_f maximum frequency

```

Each rank will process specific files (depending on its rank number) inside a directory tree and will eventually write a `.txt` file as a checkpoint indicating that all files assigned to such a rank have already been processed in such a directory.
If the run has to be re-started the rank will try to read the checkpoint files in order to avoid starting from scratch.
If the rank finds a checkpoint file corresponding to its rank numer in a directory, that means that all the audio files have already been processed by such a rank in that directory. Therefore such a rank will jump the processing of such audio files.

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








For a HPC cluster, on a node with shared memory, on 8 GPUs, we use the following command on ThetaGPU (a compute cluster at Argonne National Laboratory)

This command submit the job to run during 10 minutes.

`qsub -n 1 -q full-node -t 10 -A MyProjectName ./my_script_to_run.sh`

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
 11 singularity exec --nv -B $spectrogram_images_path:/Spectrogram_Images,$model_path:/Model $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $train_dino_path --arch vit_small --data_path /Spectrogram_Images --output_dir /Model --use_fp16 false --epo    chs 15 --saveckp_freq 1 --warmup_epochs 5 --batch_size_per_gpu 64 --local_crops_number 20 --lr 0.00005
 12 
 13 
 14 
 
```











## Joint Embedding Inference on shared memory

The following command runs an inference instance on a shared memory system on a single GPU.
This is generally used in a local machine


`python -m torch.distributed.launch --nproc_per_node=1 inference_dino.py --arch vit_small --data_path /path/to/your/images/ --dump_features /path/where/the/features/will/be/dumpped/ --pretrained_weights /path/to/the/pretrained/weights/checkpoint.pth --batch_size_per_gpu some integer --image_size 512 512 --inference_up_to 1000`







For a HPC cluster, on a node with shared memory, on 8 GPUs, we use the following command on ThetaGPU (a compute cluster at Argonne National Laboratory)

This command submit the job to run during 10 minutes.

`qsub -n 1 -q full-node -t 10 -A MyProjectName ./my_script_to_run.sh`

where `my_script_to_run.sh` is

```
  1 #!/bin/sh
  2 
  3 # Common paths
  4 spectrogram_images_path='/path/to/the/spectrograms'
  5 singularity_image_path='/path/to/the/singularity/image.sif'
  6 dino_path='/path/to/the/dino/python/script'
  7 inference_dino_path='/path/to/the/inference_dino.py'
  8 model_path='/path/to/where/the/models/were/saved'
  9 features_path='/path/to/where/the/features/will/be/saved'
 10 
 11 cd $dino_path
 12 singularity exec --nv -B $spectrogram_images_path:/Spectrogram_Images,$model_path:/Model,$features_path:/Features $singularity_image_path python -m torch.distributed.launch --nproc_per_node=8 $inference    _dino_path --arch vit_small --data_path /Spectrogram_Images --dump_features /Features --pretrained_weights /Model/checkpoint0014.pth --batch_size_per_gpu 64 --image_size 349 475
 13 
 
```


After running this command the features will be saved in `--demp_features` path.
The features are composed by 3 files:

```
att_map.pth
feat.pth
file_name.pth
```



- `att_map.pth` contains all the attentional maps corresponding to each image.
- `feat.pth` contains an output feature vector from DINO per input image.
- `file_name.pth` contains an array with information about the input file names.



