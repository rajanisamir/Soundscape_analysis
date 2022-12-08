import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from maad import sound
from maad import util

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




def reformat_all_files_in_dir(path, extention):
    for File in os.listdir(path):
        filename = os.fsdecode(File)
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

    # You should change 'test' to your preferred folder.
    CHECK_FOLDER = os.path.isdir(outputdirpath)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        Path(outputdirpath).mkdir(parents=True, exist_ok=True)

    save_figures(outputdirpath, Path(audiofilepath).stem, figures)
    reformat_all_files_in_dir(outputdirpath, 'png')







