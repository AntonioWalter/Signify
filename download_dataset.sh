#!/bin/bash
# Script con retry automatico per scaricare il dataset ASL Citizen da Kaggle
# In caso di connessione interrotta, riprende automaticamente da dove si era fermato

# Attiva il virtual environment
cd /Users/antoniowalterdefusco/Documents/Project/Signify
source venv/bin/activate

MAX_RETRIES=20
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_RETRIES ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "============================================"
    echo "Tentativo $ATTEMPT / $MAX_RETRIES"
    echo "============================================"

    python -c "
import kagglehub, shutil, os, sys

path = kagglehub.dataset_download('abd0kamel/asl-citizen')
print('Download completato! Path:', path)

dest = 'data/raw/asl_citizen'
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(path, dest)
print('Copiato in:', dest)
print('Contenuto:', os.listdir(dest))
sys.exit(0)
"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "============================================"
        echo "Download completato con successo!"
        echo "============================================"
        exit 0
    else
        echo "Connessione interrotta. Riprovo tra 10 secondi..."
        sleep 10
    fi
done

echo "Raggiunto il numero massimo di tentativi ($MAX_RETRIES)."
exit 1
