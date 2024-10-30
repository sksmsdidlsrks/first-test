from paddleocr import PaddleOCR
import cv2

def extract_text(image_path):
    # PaddleOCR 리더 생성 (한국어 지원)
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 한국어 지원

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 텍스트 추출
    result = ocr.ocr(image_path, cls=True)

    # 결과 출력
    for line in result:
        for word_info in line:
            print(f"텍스트: {word_info[1][0]}, 확률: {word_info[1][1]:.2f}")

# 사용 예
image_path = 'img\\ticket4.jpg'  # 실제 이미지 경로로 변경
extract_text(image_path)