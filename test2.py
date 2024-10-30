import easyocr
import cv2
import matplotlib.pyplot as plt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# EasyOCR 리더 초기화 (영어와 한국어 지원)
reader = easyocr.Reader(['ko', 'en'])

# 메인 함수
def main(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
        return

    # 텍스트 인식
    result = reader.readtext(image)

    # 인식된 텍스트 이미지 위에 표시
    extracted_texts = []
    for (bbox, text, prob) in result:
        # 텍스트의 위치
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))

        # 텍스트 표시
        cv2.putText(image, text, (top_left[0], top_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 바운딩 박스 표시
        cv2.rectangle(image, (top_left[0], top_left[1]), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 2)

        # 인식된 텍스트 출력
        extracted_texts.append(text)
        print(f"인식된 텍스트: {text} (신뢰도: {prob:.2f})")

    # 모든 텍스트를 하나의 문자열로 결합
    full_text = " ".join(extracted_texts)

    # 텍스트 요약
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # 요약할 문장 수

    print("\n요약된 텍스트:")
    for sentence in summary:
        print(sentence)

    # BGR 이미지를 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Matplotlib을 사용하여 이미지 표시
    plt.imshow(image_rgb)
    plt.axis('off')  # 축 제거
    plt.title("인식된 텍스트")
    plt.show()

# 이미지 파일 경로 설정
image_path = "img\\test2.png"  # 여기에 이미지 경로를 입력하세요

# 메인 함수 호출
if __name__ == "__main__":
    main(image_path)
