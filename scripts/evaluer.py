import torch
from models.mon_modele import charger_modele
from utils.preprocessing import preprocess_image

def evaluer_image(image_path):
    """Évaluer une image et prédire sa classe."""
    modele = charger_modele()
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = modele(image)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Vérifie si le modèle a une propriété id2label pour traduire les indices en labels
    if hasattr(modele.config, "id2label"):
        num_labels = len(modele.config.id2label)
        k = min(5, num_labels)  # S'assure que k ne dépasse pas le nombre de labels
        topk_prob, topk_catid = torch.topk(predictions, k)
        for i in range(topk_prob.size(1)):
            print(f"{modele.config.id2label[topk_catid[0][i].item()]}: {topk_prob[0][i].item():.2f}")
    else:
        print("Le modèle n'a pas d'attribut 'id2label'. Impossible de traduire les indices de classes en labels.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        evaluer_image(image_path)
    else:
        print("Veuillez fournir un chemin d'image en argument.")
