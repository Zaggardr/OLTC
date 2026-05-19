#!/bin/bash
echo "Installation des dépendances..."
pip install -r requirements.txt
echo ""
echo "Lancement du dashboard OLTC sur http://localhost:8501"
streamlit run app.py --server.port 8501 --server.address localhost --server.headless false
