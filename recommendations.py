import numpy as np


def generate_recommendation(latest, nasa_latest=None):
    soil = float(latest.get('Soil Moisture', np.nan))
    temp = float(latest.get('Temperature', np.nan))
    ndvi = float(latest.get('NDVI', np.nan))
    precip = float(latest.get('rainfall', np.nan))

    if nasa_latest is not None:
        precip = float(nasa_latest.get('rainfall', precip))

    messages = []

    if soil < 30:
        messages.append("Arrosage recommandé : humidité du sol faible.")
    elif soil > 70 and precip > 5:
        messages.append("Réduire l'irrigation : sol déjà humide et pluie récente.")
    elif soil > 80:
        messages.append("Sur-irrigation possible : vérifier l'évacuation.")
    else:
        messages.append("Humidité du sol dans une plage acceptable.")

    if temp > 34 and soil < 40:
        messages.append("Chaleur élevée et sol sec : irriguer tôt le matin.")
    elif temp < 10:
        messages.append("Température basse : limiter les interventions mécaniques.")

    if precip >= 8:
        messages.append("Pluie importante détectée : arrêter l'irrigation et contrôler le drainage.")
    elif precip >= 1:
        messages.append("Prévision de précipitations légère à modérée : adapter la planification d'irrigation.")

    if ndvi < 0.35:
        messages.append("NDVI bas : croissance végétale faible, surveiller la fertilisation et l'état des plantes.")
    elif ndvi > 0.7:
        messages.append("NDVI élevé : bonne végétation, continuer le suivi normal.")
    else:
        messages.append("NDVI moyen : conditions de culture stables, maintenir les mesures actuelles.")

    return ' '.join(messages)
