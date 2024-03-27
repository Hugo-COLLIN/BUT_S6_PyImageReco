from transformers import ViTForImageClassification

def charger_modele():
    """Charger un modèle pré-entraîné pour la classification d'images."""
    modele = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    return modele
