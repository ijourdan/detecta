## detecta
Detecta autos, extrae la imágen y los almacena en ./out

**python3 auto -d <# de dispositivo>**

Por lo general el dispositivo es 0, cuando la cámara está integrada o es el único dispositivo.
El dispositivo 0 está por default. En ese caso, es suficiente

python3 auto

ó

./auto

si estuviera auto.py con los permisos adecuado

Si la cámara que se quiere emplear para capturar no fuera el dispositivo 0, probar con otros número

python3 auto -d 1
python3 auto --disp 1
./auto -d 1
