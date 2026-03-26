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
resultados_ia = None

model_path = r'C:\Users\pedro\OneDrive\Documentos\handtracking\gesture_recognizer.task'
base_options = BaseOptions(model_asset_path=model_path)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    global resultados_ia
    resultados_ia = result
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=r'C:\Users\pedro\OneDrive\Documentos\handtracking\gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    camera = cv2.VideoCapture(0)
    while True:
        sucesso, frame = camera.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = int(time.time() * 1000)
        altura, largura, _ = frame.shape
        if resultados_ia != None:
            for mao in resultados_ia.hand_landmarks:
                for ponto in mao:
                    pixel_x = int(ponto.x * largura)
                    pixel_y = int(ponto.y * altura)
                    cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)
                
        cv2.imshow('Rastreador', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, timestamp)
    camera.release()
    cv2.destroyAllWindows()
    