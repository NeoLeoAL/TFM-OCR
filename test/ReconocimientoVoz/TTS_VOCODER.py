from re import X
import soundfile as sf
import pygame as pg
import pyworld as pw
from gtts import gTTS
from tempfile import TemporaryFile
from pydub import AudioSegment 
import os
import matplotlib.pyplot as plt

tts = gTTS(text='Hola. ¿Cómo estás?', lang='es')
ficheroMP3 = 'IA_VOC/saludo.mp3'
tts.save(ficheroMP3)

sound = AudioSegment.from_mp3(ficheroMP3)
ficheroWAV = 'IA_VOC/saludo.wav'
sound.export(ficheroWAV, format='wav')

ficheroVOC = 'IA_VOC/saludoVOC.wav'

velocidad = 2
timbre = 3.5
volumen = 1

x, fs = sf.read(ficheroWAV)
print(f'Número de muestras de audio: {len(x)}')

""" plt.plot(x)
plt.ylabel('Señal de voz')
plt.show() """

f0, sp, ap = pw.wav2world(x, fs)
print(f'Número de valores calculados de f0: {len(f0)} y ap: {len(ap)}')
print(f'Número de muestras usadas para calcular f0 y ap: {len(x)/len(f0)}')

""" plt.plot(f0)
plt.ylabel('Pitch')
plt.show()

plt.plot(ap)
plt.ylabel('Aperiodicidad')
plt.show()

plt.plot(sp)
plt.ylabel('Espectograma')
plt.show() """

yy = pw.synthesize(f0/timbre, sp/volumen, ap, fs/velocidad, pw.default_frame_period)
sf.write(ficheroVOC, yy, fs)

pg.mixer.init()
pg.mixer.music.load(ficheroVOC)
print('Reproduciendo ' + ficheroVOC)

pg.mixer.music.play()
while pg.mixer.music.get_busy():
    pg.time.Clock().tick(10)
    
pg.mixer.quit()

os.remove(ficheroVOC)