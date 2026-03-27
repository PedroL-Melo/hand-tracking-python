import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
resultados_ia = None

model_path = r'C:\Users\pedro\OneDrive\Documentos\handtracking\gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    global resultados_ia
    resultados_ia = result
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=r'C:\Users\pedro\OneDrive\Documentos\handtracking\gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2)
with GestureRecognizer.create_from_options(options) as recognizer:
    camera = cv2.VideoCapture(0)
    MAPA_DOS_OSSOS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Dedão
        (0, 5), (5, 6), (6, 7), (7, 8),       # Indicador
        (5, 9), (9, 10), (10, 11), (11, 12),  # Dedo Médio
        (9, 13), (13, 14), (14, 15), (15, 16),# Anelar
        (13, 17), (17, 18), (18, 19), (19, 20),# Minguinho
        (0, 17)                               # Fechando a palma da mão
    ]
    foto = cv2.imread(r'C:\Users\pedro\OneDrive\Documentos\handtracking\foto\images.jpeg')
    while True:
        sucesso, frame = camera.read()
        frame =  cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(time.time() * 1000)
        altura, largura, _ = frame.shape
        foto_tela_cheia = cv2.resize(foto, (largura, altura))
        if resultados_ia != None:
            if resultados_ia.gestures:
                quant_maos = len(resultados_ia.gestures)
                if quant_maos == 2:
                    gesto_1 = resultados_ia.gestures[0][0].category_name
                    gesto_2 = resultados_ia.gestures[1][0].category_name
                    if gesto_1 == 'Open_Palm' and gesto_1 == gesto_2:
                        frame[0:altura, 0:largura] = foto_tela_cheia
            for mao in resultados_ia.hand_landmarks:
                pontos_maos = []
                for ponto in mao:
                    pixel_x = int(ponto.x * largura)
                    pixel_y = int(ponto.y * altura)
                    pontos_maos.append((pixel_x, pixel_y))
                    cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)
                for conexao in MAPA_DOS_OSSOS:
                    ponto_a = pontos_maos[conexao[0]]
                    ponto_b = pontos_maos[conexao[1]]
                    cv2.line(frame, ponto_a, ponto_b, (255, 0, 0), 2)
        cv2.imshow('Rastreador', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        recognizer.recognize_async(mp_image, timestamp)
    camera.release()
    cv2.destroyAllWindows()