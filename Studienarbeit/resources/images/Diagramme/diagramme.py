import matplotlib.pyplot as plt
import numpy as np

# F1-Scores aus der Tabelle extrahieren
plt.rcParams['font.family'] = 'Arial'
models = ["Custom CNN", "Mobilenet v3", "Mobilenet v1"]
f1_scores_bad = [0.98, 0.44, 0.83]

x = np.arange(len(models))  # Position der Gruppen auf der x-Achse

# Diagramm erstellen
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35  # Breite der Balken

# Balken für "Bad" und "Good" F1-Scores
bars1 = ax.bar(x, f1_scores_bad, bar_width, label="gemittelter F1", color='royalblue')


# Achsenbeschriftungen und Titel
ax.set_xlabel("Modelle",fontsize=12, fontweight='bold')
ax.set_ylabel("F1-Score",fontsize=12, fontweight='bold')
ax.set_title("F1-Scores der Modelle", fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Werte über die Balken schreiben
for bars in [bars1]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset nach oben
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Diagramm anzeigen
plt.ylim(0, 1.1)  # Skala anpassen für bessere Sichtbarkeit
#plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()
