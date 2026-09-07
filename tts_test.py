import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "In 1241, a Mongol army under general Subutai approached the Sajo River in Hungary, facing a Hungarian-led force nearly three times their size."

wav = model.generate(text)
ta.save("narration_test.wav", wav, model.sr)
print("Saved narration_test.wav")