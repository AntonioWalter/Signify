"""
Analisi completa del dataset Dataset_Keypoints_Train.

Produce grafici e statistiche su:
1. Distribuzione del numero di campioni per segno (classe)
2. Distribuzione della lunghezza delle sequenze (numero di frame)
3. Rilevamento di landmark mancanti (frame completamente a zero)
4. Distribuzione per signer (firmatario)
5. Statistiche aggregate per pianificare il bilanciamento
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# --- Configurazione ---
DATASET_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/data/processed/Dataset_Keypoints_Train"
OUTPUT_DIR = "/Users/antoniowalterdefusco/Documents/Project/Signify/docs/latex/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stile grafici
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# --- 1. Raccolta dati ---
print("Scansione dataset in corso...")

samples_per_class = {}
frame_lengths = []
frame_lengths_per_class = defaultdict(list)
zero_pose_frames = 0
zero_lh_frames = 0
zero_rh_frames = 0
total_frames = 0
signer_counter = Counter()
files_with_all_zero_components = []
total_files = 0

glosses = sorted(os.listdir(DATASET_DIR))
glosses = [g for g in glosses if os.path.isdir(os.path.join(DATASET_DIR, g))]

for gloss in glosses:
    gloss_dir = os.path.join(DATASET_DIR, gloss)
    npy_files = [f for f in os.listdir(gloss_dir) if f.endswith('.npy')]
    samples_per_class[gloss] = len(npy_files)

    for npy_file in npy_files:
        total_files += 1
        filepath = os.path.join(gloss_dir, npy_file)

        # Estrai signer ID dal nome file (formato: signerID-GLOSS.npy)
        signer_id = npy_file.split('-')[0] if '-' in npy_file else 'unknown'
        signer_counter[signer_id] += 1

        try:
            data = np.load(filepath)
            n_frames = data.shape[0]
            frame_lengths.append(n_frames)
            frame_lengths_per_class[gloss].append(n_frames)

            # Analisi landmark mancanti per frame
            for frame in data:
                total_frames += 1
                pose = frame[:132]
                lh = frame[132:195]
                rh = frame[195:258]

                if np.all(pose == 0):
                    zero_pose_frames += 1
                if np.all(lh == 0):
                    zero_lh_frames += 1
                if np.all(rh == 0):
                    zero_rh_frames += 1

                if np.all(pose == 0) and np.all(lh == 0) and np.all(rh == 0):
                    pass  # frame completamente vuoto

        except Exception as e:
            print(f"Errore nel caricamento di {filepath}: {e}")

    if total_files % 1000 == 0:
        print(f"  Analizzati {total_files} file...")

print(f"Scansione completata: {total_files} file analizzati.")

# --- 2. Statistiche aggregate ---
counts = list(samples_per_class.values())
counts_arr = np.array(counts)

print("\n" + "=" * 60)
print("STATISTICHE DISTRIBUZIONE CLASSI")
print("=" * 60)
print(f"Numero totale di classi (segni): {len(samples_per_class)}")
print(f"Numero totale di campioni: {sum(counts)}")
print(f"Media campioni per classe: {np.mean(counts_arr):.1f}")
print(f"Mediana campioni per classe: {np.median(counts_arr):.1f}")
print(f"Deviazione standard: {np.std(counts_arr):.1f}")
print(f"Minimo campioni: {np.min(counts_arr)} ({sum(1 for c in counts if c == np.min(counts_arr))} classi)")
print(f"Massimo campioni: {np.max(counts_arr)}")
print(f"Classi con ≤ 5 campioni: {sum(1 for c in counts if c <= 5)}")
print(f"Classi con ≤ 10 campioni: {sum(1 for c in counts if c <= 10)}")
print(f"Classi con > 20 campioni: {sum(1 for c in counts if c > 20)}")

# Top/bottom classes
sorted_classes = sorted(samples_per_class.items(), key=lambda x: x[1])
print(f"\n5 classi con MENO campioni:")
for name, count in sorted_classes[:5]:
    print(f"  {name}: {count}")
print(f"\n5 classi con PIÙ campioni:")
for name, count in sorted_classes[-5:]:
    print(f"  {name}: {count}")

print("\n" + "=" * 60)
print("STATISTICHE LUNGHEZZA SEQUENZE (FRAME)")
print("=" * 60)
frame_arr = np.array(frame_lengths)
print(f"Media frame per video: {np.mean(frame_arr):.1f}")
print(f"Mediana frame per video: {np.median(frame_arr):.1f}")
print(f"Deviazione standard: {np.std(frame_arr):.1f}")
print(f"Minimo frame: {np.min(frame_arr)}")
print(f"Massimo frame: {np.max(frame_arr)}")
print(f"Percentile 25°: {np.percentile(frame_arr, 25):.0f}")
print(f"Percentile 75°: {np.percentile(frame_arr, 75):.0f}")
print(f"Percentile 95°: {np.percentile(frame_arr, 95):.0f}")

print("\n" + "=" * 60)
print("STATISTICHE LANDMARK MANCANTI")
print("=" * 60)
print(f"Frame totali analizzati: {total_frames:,}")
print(f"Frame con pose a zero: {zero_pose_frames:,} ({100*zero_pose_frames/total_frames:.2f}%)")
print(f"Frame con mano SX a zero: {zero_lh_frames:,} ({100*zero_lh_frames/total_frames:.2f}%)")
print(f"Frame con mano DX a zero: {zero_rh_frames:,} ({100*zero_rh_frames/total_frames:.2f}%)")

print("\n" + "=" * 60)
print("STATISTICHE SIGNER")
print("=" * 60)
print(f"Signer unici: {len(signer_counter)}")
signer_counts = sorted(signer_counter.values(), reverse=True)
print(f"Media video per signer: {np.mean(signer_counts):.1f}")
print(f"Max video per signer: {max(signer_counts)}")
print(f"Min video per signer: {min(signer_counts)}")

# --- 3. Grafici ---
print("\nGenerazione grafici...")

# GRAFICO 1: Distribuzione campioni per classe (istogramma)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(counts_arr, bins=50, color='#2E6B9E', edgecolor='white', alpha=0.85)
ax.set_xlabel('Numero di Campioni per Classe')
ax.set_ylabel('Numero di Classi')
ax.set_title('Distribuzione del Numero di Campioni per Segno')
ax.axvline(np.mean(counts_arr), color='#E74C3C', linestyle='--', linewidth=2, label=f'Media: {np.mean(counts_arr):.1f}')
ax.axvline(np.median(counts_arr), color='#F39C12', linestyle='--', linewidth=2, label=f'Mediana: {np.median(counts_arr):.1f}')
ax.legend(fontsize=11)
plt.savefig(os.path.join(OUTPUT_DIR, 'distribuzione_campioni_per_classe.png'))
plt.close()
print("  ✓ distribuzione_campioni_per_classe.png")

# GRAFICO 2: Top 30 e Bottom 30 classi (barplot)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Bottom 30
bottom_30 = sorted_classes[:30]
names_b = [x[0] for x in bottom_30]
vals_b = [x[1] for x in bottom_30]
axes[0].barh(names_b, vals_b, color='#E74C3C', edgecolor='white')
axes[0].set_xlabel('Numero di Campioni')
axes[0].set_title('30 Classi con MENO Campioni')
axes[0].tick_params(axis='y', labelsize=8)
axes[0].invert_yaxis()

# Top 30
top_30 = sorted_classes[-30:]
names_t = [x[0] for x in top_30]
vals_t = [x[1] for x in top_30]
axes[1].barh(names_t, vals_t, color='#27AE60', edgecolor='white')
axes[1].set_xlabel('Numero di Campioni')
axes[1].set_title('30 Classi con PIÙ Campioni')
axes[1].tick_params(axis='y', labelsize=8)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_bottom_classi.png'))
plt.close()
print("  ✓ top_bottom_classi.png")

# GRAFICO 3: Distribuzione lunghezza sequenze
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(frame_arr, bins=80, color='#8E44AD', edgecolor='white', alpha=0.85)
ax.set_xlabel('Numero di Frame per Video')
ax.set_ylabel('Numero di Video')
ax.set_title('Distribuzione della Lunghezza delle Sequenze')
ax.axvline(np.mean(frame_arr), color='#E74C3C', linestyle='--', linewidth=2, label=f'Media: {np.mean(frame_arr):.1f}')
ax.axvline(np.percentile(frame_arr, 95), color='#F39C12', linestyle=':', linewidth=2, label=f'95° percentile: {np.percentile(frame_arr, 95):.0f}')
ax.legend(fontsize=11)
plt.savefig(os.path.join(OUTPUT_DIR, 'distribuzione_frame.png'))
plt.close()
print("  ✓ distribuzione_frame.png")

# GRAFICO 4: Landmark mancanti (barplot componenti)
fig, ax = plt.subplots(figsize=(10, 6))
components = ['Pose\n(Corpo)', 'Mano\nSinistra', 'Mano\nDestra']
zero_pcts = [
    100 * zero_pose_frames / total_frames,
    100 * zero_lh_frames / total_frames,
    100 * zero_rh_frames / total_frames
]
colors = ['#2E6B9E', '#E74C3C', '#27AE60']
bars = ax.bar(components, zero_pcts, color=colors, edgecolor='white', width=0.5)
ax.set_ylabel('% Frame con Componente a Zero')
ax.set_title('Percentuale di Frame con Landmark Mancanti')
for bar, pct in zip(bars, zero_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
            f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
plt.savefig(os.path.join(OUTPUT_DIR, 'landmark_mancanti.png'))
plt.close()
print("  ✓ landmark_mancanti.png")

# GRAFICO 5: Boxplot campioni per classe (compatto)
fig, ax = plt.subplots(figsize=(10, 3))
ax.boxplot(counts_arr, vert=False, widths=0.6,
           boxprops=dict(color='#2E6B9E'), medianprops=dict(color='#E74C3C', linewidth=2),
           whiskerprops=dict(color='#2E6B9E'), capprops=dict(color='#2E6B9E'),
           flierprops=dict(marker='o', markerfacecolor='#F39C12', markersize=4, alpha=0.5))
ax.set_xlabel('Campioni per Classe')
ax.set_title('Distribuzione Campioni per Classe (Boxplot)')
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_campioni.png'))
plt.close()
print("  ✓ boxplot_campioni.png")

# GRAFICO 6: Distribuzione cumulativa (CDF) dei campioni per classe
fig, ax = plt.subplots(figsize=(10, 6))
sorted_counts = np.sort(counts_arr)
cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
ax.plot(sorted_counts, cdf, color='#2E6B9E', linewidth=2)
ax.set_xlabel('Numero di Campioni per Classe')
ax.set_ylabel('% Cumulativa di Classi')
ax.set_title('Distribuzione Cumulativa del Numero di Campioni')
ax.axhline(50, color='#E74C3C', linestyle='--', alpha=0.5, label='50%')
ax.axhline(90, color='#F39C12', linestyle='--', alpha=0.5, label='90%')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'cdf_campioni.png'))
plt.close()
print("  ✓ cdf_campioni.png")

print("\n✅ Tutti i grafici salvati in:", OUTPUT_DIR)
print("=" * 60)
