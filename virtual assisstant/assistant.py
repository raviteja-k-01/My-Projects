# import required modules
import os
import datetime
import pyttsx3
import speech_recognition as sr
import webbrowser

# Create engine for text to speech 
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    print("assistant:", text)
    engine.say(text)
    engine.runAndWait()

# command taking function 
def take_command():
    listener = sr.Recognizer() 
    with sr.Microphone() as source:
        print("Listening ....")
        audio = listener.listen(source)
        try:
            command = listener.recognize_google(audio)
            command = command.lower()
            print("You Said:", command)
            return command
        except:
            return ""

# create run assistant function 
def run_assistant():
    command = take_command()
    
    if "time" in command:
        time = datetime.datetime.now().strftime("%I:%M %p")   # fixed
        speak(f"The current time is {time}")
    
    elif "date" in command:
        today = datetime.date.today()
        speak(f"Today's date is {today}")
    
    elif "open notepad" in command:
        speak("Opening Notepad")
        os.system('notepad')
    
    elif "open youtube" in command:   # fixed
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com/")
    
    elif "hey siri" in command:
        query = command.replace("hey siri", "").strip()
        
        if query:
            url = f"https://www.google.com/search?q={query}"  # fixed
            speak(f"Searching for {query}")
            webbrowser.open(url)
        else:
            speak("I am here to assist you")
    
    elif "bye" in command or "stop" in command:
        speak("Okay, bye bye, see you soon")
        exit()
    
    else:
        speak("I am here to assist you. Ask something like 'open YouTube', 'open notepad', or 'time'.")

# main loop
if __name__ == "__main__":   # fixed
    speak("Hey hi, I am here to assist you")
    while True:
        run_assistant()
