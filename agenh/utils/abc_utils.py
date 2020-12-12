from midi2audio import FluidSynth
import music21
from pathlib import Path
import requests
import os
from string import ascii_letters
import subprocess
import torchaudio
from typing import List

def abc2midi(abc_file: Path, midi_file: Path) -> None:
    command = f'abc2midi {abc_file} -o {midi_file}'
    subprocess.run(command.split(), timeout=2, check=True)

def midi2wav(midi_file: Path, wav_file: Path) -> None:
    FluidSynth(sound_font='GeneralUser GS 1.471/GeneralUser GS v1.471.sf2', sample_rate=8000).midi_to_audio(midi_file, wav_file)

def get_wav_from_abc(filename):
    print('FILENAME: ', filename)
    midi_f = '{}.midi'.format(filename)
    wav_f = '{}.wav'.format(filename)
    abc2midi(filename, midi_f)
    midi2wav(midi_f, wav_f)
    w, sr = torchaudio.load(wav_f)
    os.remove(midi_f)
    os.remove(wav_f)
    print('WAV SHAPE', wav[0].shape)
    return wav[0]

