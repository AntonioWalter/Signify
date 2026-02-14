import cv2
import numpy as np
import os
import mediapipe as mp
import zipfile
import shutil
import pandas as pd
from multiprocessing import Pool, cpu_count, Lock
from tqdm import tqdm
import logging
from datetime import datetime

# --- CONFIGURAZIONE ---
ZIP_PATH = '/Users/antoniowalterdefusco/Documents/Project/ML/TraduttoreASL/PointExtractor/ASLCitizenArchive.zip'
OUTPUT_BASE = "Dataset_Keypoints_Train"
TEMP_DIR = "parallel_temp"
TRAIN_CSV_PATH = 'ASL_Citizen/splits/train.csv'

# Setup Logging dettagliato
logging.basicConfig(
    filename=f'extraction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_keypoints(results):
    """Estrae i 258 punti matematici."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])


def process_video_worker(args):
    """Funzione eseguita in parallelo da ogni core."""
    video_filename, label, worker_id = args
    zip_video_path = f"ASL_Citizen/videos/{video_filename}"

    # Cartella di output specifica per la parola
    action_dir = os.path.join(OUTPUT_BASE, str(label).upper().strip())
    os.makedirs(action_dir, exist_ok=True)

    output_file = os.path.join(action_dir, video_filename.split('.')[0] + ".npy")

    # Se il file esiste già, saltiamo (Resume capability)
    if os.path.exists(output_file):
        return f"SKIP: {video_filename}"

    # Ogni worker deve aprire lo ZIP per leggere il suo file (thread-safe)
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            # Estrazione temporanea specifica per il worker
            temp_video_path = os.path.join(TEMP_DIR, f"worker_{worker_id}_{video_filename}")
            with z.open(zip_video_path) as source, open(temp_video_path, "wb") as target:
                shutil.copyfileobj(source, target)

        # Inizializzazione MediaPipe all'interno del worker (Headless)
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
            cap = cv2.VideoCapture(temp_video_path)
            sequence = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Pre-processing veloce
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = holistic.process(frame)
                sequence.append(extract_keypoints(results))
                frame_count += 1

            cap.release()

            if len(sequence) > 0:
                np.save(output_file, np.array(sequence))
                logging.info(f"SUCCESS: {video_filename} | Label: {label} | Frames: {frame_count}")
            else:
                logging.warning(f"EMPTY: {video_filename} non ha prodotto frame validi.")

        # Pulizia video temporaneo
        os.remove(temp_video_path)
        return f"OK: {video_filename}"

    except Exception as e:
        logging.error(f"ERROR: {video_filename} | {str(e)}")
        return f"FAIL: {video_filename} | {str(e)}"


def main():
    start_time = datetime.now()
    print(f"🚀 Turbo Extractor avviato alle {start_time.strftime('%H:%M:%S')}")

    # Caricamento CSV
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open(TRAIN_CSV_PATH) as f:
            df = pd.read_csv(f)

    # Identificazione colonne (usando quelle che abbiamo trovato prima)
    col_video = 'Video file'
    col_label = 'Gloss'

    tasks = [(row[col_video], row[col_label], i) for i, row in df.iterrows()]
    total_tasks = len(tasks)

    print(f"📦 Dataset: {total_tasks} video. Core rilevati: {cpu_count()}")
    print(f"📂 I log dettagliati sono salvati in: extraction_log.log")

    os.makedirs(TEMP_DIR, exist_ok=True)

    # Configurazione Pool di processi (M1 Pro ha 8-10 core, ne usiamo N-1 per stabilità)
    num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        # tqdm crea la barra di progresso
        for result in tqdm(pool.imap_unordered(process_video_worker, tasks), total=total_tasks,
                           desc="Estrazione Punti"):
            pass  # Il lavoro grosso avviene nel worker

    # Pulizia finale
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✅ COMPLETATO in {duration}")
    print(f"📁 Dati salvati in: {OUTPUT_BASE}")


if __name__ == "__main__":
    # Necessario per multiprocessing su macOS
    main()