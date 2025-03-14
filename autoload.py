import pyautogui
import time

# Wartezeit, um das Skript starten zu lassen (optional)
time.sleep(2)

counter = 0

# Liste der Klickpositionen (x, y)
click_positions = [(1224, 763), (240, 246), (788, 940)]
while counter < 10:
	print(counter)
	for pos in click_positions:
    		pyautogui.click(pos[0], pos[1])  # Klick an der angegebenen Position
    		time.sleep(1)  # Kurze Pause zwischen den Klicks
	time.sleep(370)
	counter = counter + 1
	