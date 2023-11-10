import cv2


def readIPWriteTOFile():
    input_video_path = r"C:\Users\valen\PycharmProjects\algoritmMultimedia\video\video.mp4"

    # Путь к файлу, в который вы хотите записать видео
    output_video_path = r"C:\Users\valen\PycharmProjects\algoritmMultimedia\video\output_video.mp4"

    # Открытие исходного видеопотока для чтения
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Ошибка при открытии исходного видеофайла.")
        exit()


    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определение кодека и создание объекта VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи видео в формате MP4
    out = cv2.VideoWriter(output_video_path, fourcc, 25, (w, h))

    if not out.isOpened():
        print("Ошибка при создании файла для записи видео.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Конец видеопотока.")
            break

        # Запись кадра в выходное видео
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Закрытие видеопотоков и файлов
    cap.release()
    out.release()
    cv2.destroyAllWindows()


readIPWriteTOFile()
