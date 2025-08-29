import pyttsx3

# --- Simple text to speech ---
pyttsx3.speak("This is simple text to speech python module")

# --- Male Voice Example ---
engine = pyttsx3.init()  # object creation

# RATE
rate = engine.getProperty('rate')      # current speaking rate
print("Default Rate:", rate)           # e.g., 200
engine.setProperty('rate', 120)        # setting new rate (slower)

# VOLUME
volume = engine.getProperty('volume')  # current volume (0.0 to 1.0)
print("Default Volume:", volume)       # e.g., 1.0
engine.setProperty('volume', 0.5)      # setting volume (50%)

# VOICE (Male)
voices = engine.getProperty('voices')  # list of voices
engine.setProperty('voice', voices[0].id)  # 0 → Male (depends on OS)

engine.say("This is simple text to speech python module")
engine.say("My current speaking rate is " + str(rate))
engine.runAndWait()
engine.stop()

# Save voice to file (MP3)
engine.save_to_file('This is simple text to speech python module', 'male_test.mp3')
engine.runAndWait()

# --- Female Voice Example ---
engine = pyttsx3.init()

# RATE
rate = engine.getProperty('rate')
print("Default Rate:", rate)           # e.g., 200
engine.setProperty('rate', 120)

# VOLUME
volume = engine.getProperty('volume')
print("Default Volume:", volume)       # e.g., 1.0
engine.setProperty('volume', 0.5)

# VOICE (Female)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 → Female (depends on OS)

engine.say("This is female voice text to speech module")
engine.say("My current speaking rate is " + str(rate))
engine.runAndWait()
engine.stop()

engine.save_to_file('This is female text to speech python module', 'female_test.mp3')
engine.runAndWait()

# --- Male & Female Switch ---
engine = pyttsx3.init()

# RATE
rate = engine.getProperty('rate')
print("Default Rate:", rate)           # e.g., 200
engine.setProperty('rate', 120)

# VOLUME
volume = engine.getProperty('volume')
print("Default Volume:", volume)       # e.g., 1.0
engine.setProperty('volume', 0.5)

voices = engine.getProperty('voices')

# Male
engine.setProperty('voice', voices[0].id)
engine.say("Hello, this is a male voice")

# Female
engine.setProperty('voice', voices[1].id)
engine.say("And now I am speaking in a female voice")

engine.runAndWait()