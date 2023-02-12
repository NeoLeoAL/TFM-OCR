from googletrans import *

#print(googletrans.LANGUAGES)

translator = Translator() 
test = translator.detect('Tiu frazo estas skribita en Esperanto.')
print(test.lang)

translated = translator.translate('Tiu frazo estas skribita en Esperanto.', src=test.lang, dest='en') 
print(translated.text)