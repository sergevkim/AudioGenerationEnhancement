from midi2audio import FluidSynth
import music21
from pathlib import Path
import requests
import os
from string import ascii_letters
import subprocess
import torch
import torchaudio
from typing import List

def abc2midi(abc_file: Path, midi_file: Path) -> None:
    command = f'abc2midi {abc_file} -o {midi_file}'
    subprocess.run(command.split(), timeout=2, check=True)

def midi2wav(midi_file: Path, wav_file: Path) -> None:
    sound_font = os.environ.get('SOUND_FONT_PATH')
    FluidSynth(sound_font=sound_font, sample_rate=8000).midi_to_audio(midi_file, wav_file)

def get_wav_from_abc(path, filename):
    filen = '{}/{}'.format(path, filename)
    midi_f = '{}.midi'.format(filen)
    wav_f = '{}.wav'.format(filename)
    final_name = '/content/wavs/{}.wav'.format(filename[:-4])
    abc2midi(filen, midi_f)
    midi2wav(midi_f, wav_f)

    w, sr = torchaudio.load(wav_f)
    l = w.shape[1] // 2
    from_ = l - 1024 * 32
    to_ = l + 1024 * 32
    wav = w[0][from_:to_]
    print('WAV SHAPE:', wav.shape)
    print('F NAME:', final_name)
    torchaudio.save(final_name, src=wav, sample_rate=8000)
    #torch.save(w[0][from_:to_], final_name)
    os.remove(midi_f)
    os.remove(wav_f)
    return final_name

