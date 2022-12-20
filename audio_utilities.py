import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from maad import sound
from maad import util

from mpi4py import MPI

def get_spectrograms_from_file(path, window, min_freq, max_freq):
    s, fs = sound.load(path)
    slices = sound.wave2frames(s, int(fs*window))
    slices = np.transpose(slices)

    spectrograms = []
    for slc in slices:
        Sxx, tn, fn, ext = sound.spectrogram(slc, fs,
                                             window='hann', flims=[min_freq, max_freq],
                                             nperseg=1024, noverlap=512)

        spectrograms.append(Sxx)
        
    spectrograms = np.array(spectrograms)

    # here I am catching an error when the loaded audio file is corrupted
    if 'tn' not in locals() or 'fn' not in locals() or 'ext' not in locals():
        print('XXXXXXXXXXXXX>>>>>>>>>>>>> SOMETHING SEEMS TO BE WRONG WITH THE INPUT FILE HERE: ', path)

    return spectrograms, slices, tn, fn, ext





def get_figure_from_spectrogram(Sxx, extent, db_range=96, gain=0, log_scale=True,
                                interpolation='bilinear', **kwargs):
    """
    Build spectrogram representation and return figure.
    
    Parameters
    ----------
    Sxx : 2d ndarray
        Spectrogram computed using `maad.sound.spectrogram`.
    extent : list of scalars [left, right, bottom, top]
        Location, in data-coordinates, of the lower-left and upper-right corners.
    db_range : int or float, optional
        Range of values to display the spectrogram. The default is 96.
    gain : int or float, optional
        Gain in decibels added to the signal for display. The default is 0.
    **kwargs : matplotlib figure properties
            Other keyword arguments that are passed down to matplotlib.axes.

    Plot the spectrogram of an audio.
    
    """
    Sxx_db = util.power2dB(Sxx, db_range, gain)

    figsize = kwargs.pop(
        "figsize",
        (0.20 * (extent[3] - extent[2]) / 1000, 0.33 * (extent[1] - extent[0])),
    )

    fig = plt.figure()

    # display image
    _im = plt.imshow(Sxx_db, interpolation=interpolation, extent=extent, origin="lower", cmap='gray')

    plt.axis("tight")
    plt.axis('off')
    plt.close()

    return fig




def get_figures_from_spectrograms(spectrograms, extent):
    figures = []
    for spectrogram in spectrograms:
        figures.append(get_figure_from_spectrogram(spectrogram, extent))

    return figures



def save_figures(path, file_name, figures):
    counter=0
    for figure in figures:
        aux = os.path.join(path, file_name+'_'+str(counter)+'.png')
        figure.savefig(aux, transparent=True, bbox_inches='tight')

        counter = counter+1




def reformat_all_files_in_dir(path, originalfilename, extention):
    for File in os.listdir(path):
        filename = os.fsdecode(File)
        # It will take only the spectrogram images with original file names 
        # this is important for parallel computation
        # This generates issues in parallel computation since this function could try to read files
        # that have not been completely created yet from pair processes
        if originalfilename==Path(filename).stem[:len(originalfilename)]:
            if filename.endswith('.'+extention):
                filepath = os.path.join(path, filename)
                img = reformat_image(Image.open(filepath))
                img.save(filepath)




def reformat_image(image, margin=10):
    """
    Reformat an image to gray scale and crop borders out
    """
    
    # Size of the image in pixels (size of original image)
    width, height = image.size

    # Setting the points for cropped image
    left = margin
    top = margin
    right = width-margin
    bottom = height-margin
     
    # Cropped image of above dimension
    # (It will not change original image)
    return image.convert('L').crop((left, top, right, bottom))





def build_spectrogram_images_from_audio_file(audiofilepath, outputdirpath, window_t, min_f, max_f):
    """
    Build and save spectrogram images built from audio file
    """

    spectrograms, slices, tn, fn, ext = get_spectrograms_from_file(audiofilepath, window_t, min_f, max_f)
    figures = get_figures_from_spectrograms(spectrograms, ext)

    CHECK_FOLDER = os.path.isdir(outputdirpath)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        Path(outputdirpath).mkdir(parents=True, exist_ok=True)

    save_figures(outputdirpath, Path(audiofilepath).stem, figures)
    reformat_all_files_in_dir(outputdirpath, Path(audiofilepath).stem, 'png')






def navigate_directory_tree(size, rank, input_directory, output_directory, window_t=10, min_f=0, max_f=20000):
    """
    Build and save spectrogram images built from audio files from directory trees
    """
    checkpoint_file_name = os.path.join(output_directory, 'Checkpoint_rank_' + str(rank) + '.txt')
    if not os.path.isfile(checkpoint_file_name):
        # iterate over files in
        # that input_directory
        # and then over subdirectories recursively
        files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        directories = [d for d in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, d))]
        counter = 0
        for filename in files:
            if counter % size == rank:
                # print('From rank ', rank, ' From ', os.path.join(input_directory, filename))
                # print('From rank ', rank, ' To ', output_directory)
                build_spectrogram_images_from_audio_file(os.path.join(input_directory, filename), output_directory, window_t, min_f, max_f)

            counter = counter + 1

        for dirname in directories:
            navigate_directory_tree(size=size, rank=rank,
                                    input_directory=os.path.join(input_directory, dirname),
                                    output_directory=os.path.join(output_directory, dirname), window_t=window_t, min_f=min_f, max_f=max_f)

        os.makedirs(os.path.dirname(checkpoint_file_name), exist_ok=True)
        with open(checkpoint_file_name, 'w') as f:
            f.write('Files in this directory were completed for rank ' + str(rank))
            f.close()

        print('------>>>>>>> Checkpoint completed for directory: ' + output_directory + ' for rank ' + str(rank))
    else:
        print('------>>>>>>> Checkpoint already completed for directory: ' + output_directory + ' for rank ' + str(rank))











