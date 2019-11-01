from numpy import arange, array, abs
import time
from cv2 import waitKey, resize, VideoCapture, imwrite, destroyAllWindows
from cv2.dnn import readNetFromCaffe, blobFromImage
from imutils import resize as imuresize
from pandas import Timestamp


class Target:
    def __init__(self):
        # Parametros necesarios para graficar.
        self.cap = None
        self.h = 0  # alto de la imagen capturada
        self.w = 0  # ancho de la imagen captirada
        self.old_frame = None
        # Detecciones
        self.confthr = .5  # umbral de confiabilidad
        self.dimthr = .25  # umbral de dimensión (.5 ~50% del ancho total de la imágen)
        self.box_startXY = []
        self.box_endXY = []
        self.detections = []  # detecciones
        # radar: modelo y directorio de salida
        self.prototxt = './models/MobileNetSSD_deploy.prototxt.txt'
        self.modelo = './models/MobileNetSSD_deploy.caffemodel'
        self.dir_out = './out/'

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.IGNORE = {"background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"}

        # iniciamos el modelo
        self.net = readNetFromCaffe(self.prototxt, self.modelo)

    def video_radar(self):
        ret, self.old_frame = self.cap.read()  # leemos un frame
        return ret

    def radar(self):

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            # exitcounter = 0
            key = waitKey(1) & 0xFF
            if self.video_radar():

                frame = imuresize(self.old_frame, width=400)  # mantiene el aspect ratio

                (h, w) = frame.shape[:2]  # almacenamos el tamaño del frame
                # y normalizamos (generando un blob)
                blob = blobFromImage(resize(frame, (300, 300)),  0.007843, (300, 300), 127.5)

                self.net.setInput(blob)  # pasamos el blob a la red
                self.detections = self.net.forward()  # corremos la red

                # Inicializamos para una nueva detección.
                self.box_startXY = []
                self.box_endXY = []
                # extracción de detecciones relevantes
                for i in arange(0, self.detections.shape[2]):
                    confidence = self.detections[0, 0, i, 2]
                    if confidence >= self.confthr:
                        idx = int(self.detections[0, 0, i, 1])
                        # chequeo si pertenece a ignorados
                        if self.CLASSES[idx] in self.IGNORE:
                            continue
                        # coordenadas (x, y) respecto imagen ancho 400
                        box = self.detections[0, 0, i, 3:7] * array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        # chequeo del tamaño del objeto detectado en relación a la imagen
                        if abs(startX - endX) / 400 < self.dimthr:
                            continue
                        # coordenadas (x, y) respecto imagen original
                        self.box_startXY.append((int(self.w / 400 * startX), int(self.w / 400 * startY)))
                        self.box_endXY.append((int(self.w / 400 * endX), int(self.w / 400 * endY)))

                # Se graban detecciones
                if len(self.box_startXY) > 0:
                    ts = Timestamp.now()
                    for i in range(len(self.box_startXY)):
                        # extraemos auto
                        img_cut = self.old_frame[self.box_startXY[i][1]:self.box_endXY[i][1],
                                                 self.box_startXY[i][0]:self.box_endXY[i][0]]
                        image_name = str(ts) + '_' + str(i)
                        imwrite(self.dir_out + image_name + '.jpg', img_cut)

                # cambio manual de nivel de confianza
                if (key == ord("r")) and (self.confthr < 0.99):  # aumenta sensibilidad
                    self.confthr += .1
                    print('confthr :', self.confthr)
                if (key == ord("f")) and (self.confthr > 0.01):  # baja sensibilidad
                    self.confthr -= .1
                    print('confthr: ', self.confthr)
                # cambio de % de ancho
                if (key == ord("e")) and (self.dimthr < 0.99):  # aumenta sensibilidad
                    self.dimthr += .1
                    print('dimthr: ', self.dimthr)
                if (key == ord("d")) and (self.dimthr > 0.2):  # baja sensibilidad
                    self.dimthr -= .1
                    print('dimthr: ', self.dimthr)

            # si seleccionamos 'q' se termina.
            if key == ord("q"):  # termina
                self.close_vid()
                break

    def start(self, nomb):
        """
        Inicializa el programa. Esto implica iniciar la captura de video.
        y camptura un frame
        :param nomb: Dispositivo de captura de video.
        :return:
        """
        self.cap = VideoCapture(nomb)
        if nomb == 0:
            self.cap.set(3, 1280)
            self.cap.set(4, 1024)
            time.sleep(2.0)
            self.cap.set(15, -8.0)
        else:
            # self.cap.set(3,1280)
            # self.cap.set(4,720)
            time.sleep(2.0)
            # self.cap.set(15, -8.0)

        # Toma el primer framet
        ret, self.old_frame = self.cap.read()
        exit_counter = 0
        while ret != 1:
            if exit_counter <= 100:
                exit_counter += 1
                ret, self.old_frame = self.cap.read()
            else:
                print('Video no responde. Excedida cantidad de fallas. Cierra Video')
                self.close_vid()
                break

        # ejecutamos la adquisición de datos.
        (self.h, self.w) = self.old_frame.shape[:2]
        self.radar()

    def close_vid(self):
        """
        Simplemente libera el video.
        :return:
        """
        destroyAllWindows()
        self.cap.release()
