from tkinter import scrolledtext, messagebox
import speech_recognition as sr
import tkinter as tk
import torch
from functools import partial


def predict_emotion(model, tokenizer, device, label_mapping, phrase):
    """Predict the emotion of the input phrase."""
    inputs = tokenizer(
        phrase,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return label_mapping[predicted_class]


def start_listening():
    """Capture audio and transcribe it."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            text_box.insert(tk.END, "Listening...\n")
            audio = recognizer.listen(source)
            text_box.delete(1.0, tk.END)
            text = recognizer.recognize_google(audio)
            text_box.insert(tk.END, text)
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Could not understand the audio.")
        except sr.RequestError as e:
            messagebox.showerror("Error", f"Could not request results; {e}")


def send_text(model, tokenizer, device, label_mapping):
    """Predict emotion, display the result, and clear the input."""
    phrase = text_box.get(1.0, tk.END).strip()
    if not phrase:
        messagebox.showwarning("Warning", "Input text is empty.")
        return

    emotion = predict_emotion(model, tokenizer, device, label_mapping, phrase)
    result_label.config(text=f"Predicted Emotion: {emotion}")
    text_box.delete(1.0, tk.END)


app = tk.Tk()
app.title("Emotion Detector")
app.geometry("600x500")

instructions = tk.Label(
    app, text="Press 'Start Listening' to input text via voice.", font=("Helvetica", 12)
)
instructions.pack(pady=10)

text_box = scrolledtext.ScrolledText(
    app, wrap=tk.WORD, height=5, font=("Helvetica", 12)
)
text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(app)
button_frame.pack(pady=10)

start_button = tk.Button(
    button_frame,
    text="Start Listening",
    command=start_listening,
    font=("Helvetica", 12),
    bg="#4CAF50",
    fg="white",
)
start_button.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(
    app, text="Predicted Emotion: None", fg="blue", font=("Helvetica", 14)
)
result_label.pack(pady=20)


def run_app(model, tokenizer, device):
    label_mapping = {
        0: "joy",
        1: "sadness",
        2: "anger",
        3: "fear",
        4: "love",
        5: "surprise",
    }

    send_button = tk.Button(
        button_frame,
        text="Send",
        command=partial(send_text, model, tokenizer, device, label_mapping),
        font=("Helvetica", 12),
        bg="#2196F3",
        fg="white",
    )
    send_button.pack(side=tk.LEFT, padx=10)
    app.mainloop()
