import pytesseract
import cv2

# Укажите путь к исполняемому файлу TesseractOCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread('dataset/1.jpg')
text = pytesseract.image_to_string(img,lang='rus+eng')
print(text)