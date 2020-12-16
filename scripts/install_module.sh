#!/bin/sh

pip install -e .
rm -rf agenh.egg-info
apt-get update
apt-get install abcmidi
apt-get install fluidsynth
pip install torchaudio
pip install midi2audio
