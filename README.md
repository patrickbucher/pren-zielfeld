# PREN-Zielfelderkennung

**Achtung**: Dieses Projekt ist unter der _General Public License Version 3_ lizenziert. Wer den Code verwenden will -- und sei es auch nur auszugsweise --, muss seinen gesamten Quellcode ebenfalls unter dieser Lizenz verfügbar machen.

Dieses Python-Skript (`detect_target.py`) stellt den Abstand vom Mittelpunkt
des innersten Zielfeldes zur Bildmitte fest. Die Ausgabe des Abstands erfolgt
sowohl direkt auf dem Ausgabebild als auch auf der Kommandozeile.

Das Skript `execute` ruft das Skript für sämtliche im Verzeichnis `pics/`
enthaltenen Bilder auf und demonstriert so den Ablauf der Kamerafahrt.

## Anforderungen

- Python 3.6
    - matplotlib
    - numpy
    - opencv
- eine vernünftige Shell für das `execute`-Skript
