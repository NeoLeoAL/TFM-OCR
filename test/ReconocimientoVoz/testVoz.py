# Reconocimiento de texto a voz
import pyttsx3 
  
converter = pyttsx3.init() 

converter.setProperty('rate', 150) 
converter.setProperty('volume', 1) 

voices = converter.getProperty('voices')
#converter.setProperty('voice', voices[0].id) # Español
converter.setProperty('voice', voices[1].id) # Ingles
  
converter.say("Hello, I'm Leo")
  
converter.runAndWait() 
converter.stop()

# Reconocimiento de voz a texto
import SpeechRecognition as sr
 
r = sr.Recognizer() 
 
with sr.Microphone() as source:
    print('Speak Anything : ')
    audio = r.listen(source)
 
    try:
        text = r.recognize_google(audio)
        print('You said: {}'.format(text))
    except:
        print('Sorry could not hear')