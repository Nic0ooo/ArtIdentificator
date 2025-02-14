import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from keras.models import load_model
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

# Charger le modèle et les labels
model = load_model("./inference/typeOfArt/keras_Model.h5", compile=False)
class_names = open("./inference/typeOfArt/labels.txt", "r").readlines()

model_artist = load_model("./inference/wichArtist/keras_Model.h5", compile=False)
artist_class_names = open("./inference/wichArtist/labels.txt", "r").readlines()

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classification d'Image avec Keras")
        self.setGeometry(100, 100, 600, 700)
        
        # Activer le drag and drop
        self.setAcceptDrops(True)
        
        # Layout principal
        layout = QVBoxLayout()
        
        # Label pour afficher l'image
        self.image_label = QLabel("Aucune image sélectionnée", self)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        layout.addWidget(self.image_label)
        
        # Label pour afficher le résultat
        self.result_label = QLabel("", self)
        layout.addWidget(self.result_label)
        
        # Label pour afficher l'artiste potentiel
        self.artist_result_label = QLabel("", self)
        layout.addWidget(self.artist_result_label)
        
        # Bouton pour sélectionner l'image
        self.upload_button = QPushButton("Sélectionner une image", self)
        self.upload_button.clicked.connect(self.select_image)
        layout.addWidget(self.upload_button)
        
        self.quit_button = QPushButton("Quitter", self)
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)
        
        # Définir le layout principal
        self.setLayout(layout)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.process_image(file_path)
    
    def select_image(self):
        # Ouvrir une boîte de dialogue pour sélectionner une image
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une image", "", "Images (*.jpg *.jpeg *.png *.bmp *.gif)")
        if not file_path:
            return
        self.process_image(file_path)
    
    def process_image(self, file_path):
        try:
            # Charger et préparer l'image
            image = Image.open(file_path).convert("RGB")
            image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
            
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

            arrtist_prediction = model_artist.predict(data)
            artist_index = np.argmax(arrtist_prediction)
            artist_class_name = artist_class_names[artist_index].strip()
            artist_confidence_score = arrtist_prediction[0][artist_index]

            # Remove the number from the class name
            class_name_clean = ' '.join(class_name.split(' ')[1:])
            artist_class_name_clean = ' '.join(artist_class_name.split(' ')[1:])
            
            threshol_percent_value = f"{confidence_score * 100:.0f}%"
            artist_treshold_percent_value = f"{artist_confidence_score * 100:.0f}%"
            
            # Ajouter la classe détectée sur l'image
            draw = ImageDraw.Draw(image_resized)
            font = ImageFont.load_default()
            text = f"{class_name_clean} ({threshol_percent_value})"
            
            draw.rectangle([(0, 0), (224, 20)], fill=(0, 0, 0, 128))
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            
            # Convertir l'image annotée en QImage pour PyQt5
            image_qt = self.pil_to_qimage(image_resized)
            pixmap = QPixmap.fromImage(image_qt)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), aspectRatioMode=1))
            
            # Afficher le résultat
            
            if artist_confidence_score > 0.8:
                artist_text = f"C'est surement une oeuvre de {artist_class_name_clean}"
            elif artist_confidence_score > 0.5:
                artist_text = f"Je pense que c'est une oeuvre de {artist_class_name_clean}"
            elif artist_confidence_score > 0.3:
                artist_text = f"c'est peut être une oeuvre de {artist_class_name_clean}"
        
            
            self.result_label.setText(f"C'est une oeuvre : {class_name_clean}\nScore de confiance : {threshol_percent_value}")

            if artist_text:
                self.artist_result_label.setText(artist_text)

        except Exception as e:
            self.result_label.setText(f"Erreur lors de la classification : {str(e)}")
    
    def pil_to_qimage(self, pil_image):
        """Convertit une image PIL en QImage pour PyQt5."""
        rgb_image = pil_image.convert("RGB")
        data = rgb_image.tobytes("raw", "RGB")
        qimage = QImage(data, rgb_image.size[0], rgb_image.size[1], QImage.Format_RGB888)
        return qimage

# Lancer l'application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())