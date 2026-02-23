import os
import zipfile
import tempfile
import argparse
import numpy as np
import torch
import cv2
import mediapipe as mp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import json
import re
import sys

# Aggiungi la root del progetto al path
sys.path.append(os.getcwd())

from src.models.model import LSTMAttention

def load_model(checkpoint_path, device):
    """Carica il modello addestrato e il mapping delle classi."""
    print(f"Caricamento modello da {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Parametri dal checkpoint o hardcoded se necessario
    input_size = checkpoint.get('hyperparameters', {}).get('input_size', 258)
    hidden_size = checkpoint.get('hyperparameters', {}).get('hidden_size', 512)
    num_layers = checkpoint.get('hyperparameters', {}).get('num_layers', 2)
    dropout = checkpoint.get('hyperparameters', {}).get('dropout', 0.5)
    
    # Carica mapping classi
    idx_to_class = checkpoint['idx_to_class']
    num_classes = len(idx_to_class)
    
    model = LSTMAttention(input_size, hidden_size, num_layers, num_classes, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, idx_to_class

def process_video(video_path):
    """Estrae i keypoints da un video usando MediaPipe (simulazione Real-Time)."""
    mp_holistic = mp.solutions.holistic
    
    frames_data = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Conversione BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            # Estrazione Landmark (Pose + Left Hand + Right Hand)
            frame_row = []
            
            # Pose (33 landmark * 4 val)
            if results.pose_landmarks:
                for res in results.pose_landmarks.landmark:
                    frame_row.extend([res.x, res.y, res.z, res.visibility])
            else:
                frame_row.extend([0] * (33 * 4))
                
            # Left Hand (21 landmark * 3 val)
            if results.left_hand_landmarks:
                for res in results.left_hand_landmarks.landmark:
                    frame_row.extend([res.x, res.y, res.z])
            else:
                frame_row.extend([0] * (21 * 3))

            # Right Hand (21 landmark * 3 val)
            if results.right_hand_landmarks:
                for res in results.right_hand_landmarks.landmark:
                    frame_row.extend([res.x, res.y, res.z])
            else:
                frame_row.extend([0] * (21 * 3))
                
            frames_data.append(frame_row)
            
    cap.release()
    return np.array(frames_data)

def normalize_sequence(sequence, max_frames=30):
    """Applica lo stesso resampling uniforme usato in training."""
    num_frames = sequence.shape[0]
    num_features = sequence.shape[1]
    
    if num_frames == 0:
        return np.zeros((max_frames, num_features))
        
    # Uniform Resampling to max_frames
    original_indices = np.linspace(0, num_frames - 1, num_frames)
    target_indices = np.linspace(0, num_frames - 1, max_frames)
    
    new_sequence = np.zeros((max_frames, num_features))
    for i in range(num_features):
        new_sequence[:, i] = np.interp(target_indices, original_indices, sequence[:, i])
        
    return new_sequence

def evaluate_zip(zip_path, model, device, idx_to_class):
    y_true = []
    y_pred = []
    
    # Invert mapping: ClassName -> Index
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    
    print(f"Analisi archivio: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Trova file mp4
            all_files = z.namelist()
            mp4_files = [f for f in all_files if f.lower().endswith('.mp4')]
            
            if not mp4_files:
                print("NESSUN file .mp4 trovato nel zip!")
                return

            print(f"Trovati {len(mp4_files)} video. Inizio valutazone...")
            
            for file_name in tqdm(mp4_files, desc="Processing"):
                # Estrai etichetta dal path
                # Esempio atteso: "dataset/TEST/WORD/video.mp4" o "WORD_001.mp4"
                # Cerchiamo di dedurre la label
                
                # Strategia 1: Cartella Genitore (es. "TEST/CIAO/video.mp4" -> "CIAO")
                parts = file_name.replace('\\', '/').split('/')
                
                label_candidate = None
                
                # Se è in una sottocartella, assumiamo che quella sia la classe
                if len(parts) > 1:
                    # Controlliamo la cartella che lo contiene
                    # Spesso è data/raw/test/CLASSE/video.mp4
                    candidate = parts[-2].upper()
                    if candidate in class_to_idx:
                        label_candidate = candidate
                
                # Strategia 2: Se la cartella non è una classe nota, prova il nome file
                if label_candidate is None:
                    # Rimuovi numeri e estensioni: "CIAO_01.mp4" -> "CIAO"
                    basename = os.path.basename(file_name)
                    clean_name = re.sub(r'[\d_]+\.mp4$', '', basename, flags=re.IGNORECASE).upper()
                    if clean_name in class_to_idx:
                        label_candidate = clean_name
                
                if label_candidate is None:
                    continue # Salta se non riconosciamo la classe
                    
                true_idx = class_to_idx[label_candidate]
                
                # Estrazione temporanea del video
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        z.extract(file_name, path=tmpdir)
                        full_path = os.path.join(tmpdir, file_name)
                        
                        # Estrai Features
                        raw_seq = process_video(full_path)
                        if raw_seq is None or len(raw_seq) == 0:
                            continue
                            
                        # Normalizza
                        norm_seq = normalize_sequence(raw_seq)
                        
                        # Inferenza
                        tensor = torch.FloatTensor(norm_seq).unsqueeze(0).to(device)
                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            _, pred = torch.max(outputs, 1)
                            
                        y_true.append(true_idx)
                        y_pred.append(pred.item())
                        
                    except Exception as e:
                        print(f"Errore su {file_name}: {e}")
                        continue
    except zipfile.BadZipFile:
        print("Erroe: Il file ZIP sembra corrotto o non valido.")
        return

    if not y_true:
        print("Nessun video valido processato (label non trovate o errori estrazione).")
        return

    # Calcolo Metriche
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print(f"REPORT VALUTAZIONE ({len(y_true)} campioni)")
    print("="*50)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('zip_path', type=str)
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.zip_path):
        print("File zip non trovato.")
        exit(1)
        
    model, idx_to_class = load_model(args.model, device)
    evaluate_zip(args.zip_path, model, device, idx_to_class)
