import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Parlez maintenant...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        return text
    except sr.UnknownValueError:
        print("❌ Je n'ai pas compris.")
        return None
    except sr.RequestError:
        print("⚠️ Erreur de connexion au service vocal.")
        return None
