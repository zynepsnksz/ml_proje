#!/usr/bin/env bash
# Crop Recommendation — eğitim ve test pipeline'ı
set -euo pipefail

echo "=== Model eğitimi ==="
python -m src.models.train

echo ""
echo "=== Unit testler ==="
python -m pytest tests/ -v

echo ""
echo "=== Tamamlandı ==="
echo "Demo için: streamlit run app/streamlit_app.py"
