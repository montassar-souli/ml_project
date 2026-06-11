# -*- coding: utf-8 -*-
"""
test_pipeline.py
Exemple de test unitaire pour valider le comportement de model_pipeline.
"""
import os
import pandas as pd
import pytest
from model_pipeline import prepare_data

def test_prepare_data_missing_file():
    """Vérifie que la fonction lève bien une exception si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        prepare_data(data_path="fichier_inexistant.csv")      