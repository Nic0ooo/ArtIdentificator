import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from keras.models import load_model
from PIL import Image, ImageOps, ImageTk
import numpy as np

# Initialisation de la fenêtre principale
root = tk.Tk()
root.title("Classification d'Image avec Keras")
root.geometry("600x700")

# Charger le modèle et les labels
model = load_model("./inference/keras_Model.h5", compile=False)
class_names = open("./inference/labels.txt", "r").readlines()

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Charger et afficher l'image sélectionnée
    image = Image.open(file_path).convert("RGB")
    image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image_resized)
    image_label.config(image=image_tk)
    image_label.image = image_tk
    
    # Préparer l'image pour la prédiction
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Faire la prédiction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # Afficher les résultats
    result_label.config(text=f"Classe : {class_name}\nScore de confiance : {confidence_score:.2f}")

# Interface graphique
frame = tk.Frame(root)
frame.pack(pady=20)

upload_button = tk.Button(frame, text="Sélectionner une image", command=select_image, font=("Arial", 16))
upload_button.pack()

image_label = tk.Label(root)
image_label.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 14), justify="center")
result_label.pack(pady=20)

# Lancer la boucle principale de Tkinter
root.mainloop()