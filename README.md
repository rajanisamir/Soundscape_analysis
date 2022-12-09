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

## Joint Embedding Training on shared memory

The following command runs a training instance on a shared memory system on a single GPU

`python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_small --data_path /path/to/your/images/ --output_dir /path/to/your/output/models/ --use_fp16 false --batch_size_per_gpu number of images per batch`
