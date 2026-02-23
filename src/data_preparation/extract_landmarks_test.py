"""
Script di estrazione landmark per il TEST SET del dataset ASL Citizen.

Funziona esattamente come extract_landmarks.py ma legge il file test.csv
invece di train.csv, e salva i file in Dataset_Keypoints_Test/.

Uso:
    python -m src.data_preparation.extract_landmarks_test

I file .npy verranno salvati in:
    data/processed/Dataset_Keypoints_Test/<GLOSS>/<video_id>.npy
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import zipfile
import shutil
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
from datetime import datetime

# ============================================================
# CONFIGURAZIONE — modifica ZIP_PATH con il percorso del tuo zip
# ============================================================
ZIP_PATH = '/Users/antoniowalterdefusco/Documents/Project/ML/TraduttoreASL/PointExtractor/ASLCitizenArchive.zip'
OUTPUT_BASE = "data/processed/Dataset_Keypoints_Test"
TEMP_DIR = "data/processed/test_temp"
TEST_CSV_PATH = 'ASL_Citizen/splits/test.csv'

# Setup logging
logging.basicConfig(
    filename=f'extraction_test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_keypoints(results):
    """
    Estrae i 258 punti chiave (landmark) da un frame elaborato da MediaPipe Holistic.

    Struttura:
    - Pose:       33 landmark * 4 valori (x, y, z, visibility) = 132 valori
    - Mano sx:    21 landmark * 3 valori (x, y, z)             =  63 valori
    - Mano dx:    21 landmark * 3 valori (x, y, z)             =  63 valori
    Totale:                                                       258 valori

    Se un componente non viene rilevato, viene sostituito con zeri.
    """
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                       for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(132))
    lh = (np.array([[r.x, r.y, r.z]
                     for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[r.x, r.y, r.z]
                     for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))
    return np.concatenate([pose, lh, rh])


def process_video_worker(args):
    """
    Worker per il multiprocessing: estrae i landmark da un singolo video
    e salva il risultato come file .npy.

    Ogni worker riceve il percorso del video nello zip, la sua etichetta (gloss)
    e un ID univoco per evitare conflitti sui file temporanei.
    """
    video_filename, label, worker_id = args
    zip_video_path = f"ASL_Citizen/videos/{video_filename}"

    # Cartella di output organizzata per classe
    action_dir = os.path.join(OUTPUT_BASE, str(label).upper().strip())
    os.makedirs(action_dir, exist_ok=True)

    output_file = os.path.join(action_dir, video_filename.split('.')[0] + ".npy")

    # Skip se già estratto (permette di riprendere in caso di interruzione)
    if os.path.exists(output_file):
        return f"SKIP: {video_filename}"

    try:
        # Estrazione temporanea del video dallo zip
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            temp_video_path = os.path.join(TEMP_DIR, f"worker_{worker_id}_{video_filename}")
            with z.open(zip_video_path) as source, open(temp_video_path, "wb") as target:
                shutil.copyfileobj(source, target)

        # Elaborazione con MediaPipe Holistic
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            cap = cv2.VideoCapture(temp_video_path)
            sequence = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Conversione BGR -> RGB richiesta da MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                sequence.append(extract_keypoints(results))

            cap.release()

            if len(sequence) > 0:
                np.save(output_file, np.array(sequence))
                logging.info(f"OK: {video_filename} | Label: {label} | Frames: {len(sequence)}")
            else:
                logging.warning(f"VUOTO: {video_filename} — nessun frame valido")

        os.remove(temp_video_path)
        return f"OK: {video_filename}"

    except Exception as e:
        logging.error(f"ERRORE: {video_filename} | {e}")
        return f"FAIL: {video_filename} | {e}"


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("ESTRAZIONE LANDMARK — TEST SET")
    print(f"Avvio: {start_time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # Caricamento del CSV del test set dallo zip
    print(f"\nLettura {TEST_CSV_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        with z.open(TEST_CSV_PATH) as f:
            df = pd.read_csv(f)

    col_video = 'Video file'
    col_label = 'Gloss'

    total = len(df)
    print(f"Video nel test set: {total:,}")
    print(f"Classi uniche: {df[col_label].nunique():,}")

    # Preparazione dei task per il multiprocessing
    tasks = [(row[col_video], row[col_label], i) for i, row in df.iterrows()]

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Usiamo N-1 core per stabilità
    num_workers = max(1, cpu_count() - 1)
    print(f"Worker: {num_workers} (su {cpu_count()} core disponibili)")

    print(f"\nInizio estrazione...\n{'─' * 60}")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_video_worker, tasks),
            total=total,
            desc="Estrazione Test Set"
        ))

    # Pulizia cartella temporanea
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # Riepilogo
    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = sum(1 for r in results if r.startswith("FAIL"))

    duration = datetime.now() - start_time
    print(f"\n{'=' * 60}")
    print(f"COMPLETATO in {duration}")
    print(f"  ✓ Estratti: {ok:,}")
    print(f"  ↷ Saltati:  {skip:,}")
    print(f"  ✗ Falliti:  {fail:,}")
    print(f"Dati salvati in: {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
