import pyautogui
import time

# Wartezeit, um das Skript starten zu lassen (optional)
time.sleep(2)

# Liste der Klickpositionen (x, y)
click_positions = [(100, 200), (400, 500), (700, 800)]

for pos in click_positions:
    pyautogui.click(pos[0], pos[1])  # Klick an der angegebenen Position
    time.sleep(1)  # Kurze Pause zwischen den Klicks
