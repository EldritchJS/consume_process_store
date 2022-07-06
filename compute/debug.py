import sys, os

from tqdm import tqdm

def d_print(level, frame_info, message, l2=None):
    tqdm.write(f'[{level}] - {frame_info.filename}:{frame_info.lineno}\n\t* {message}')

    if l2:
        tqdm.write(f'\t\t{l2}')

def block_print():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

def open_print():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
