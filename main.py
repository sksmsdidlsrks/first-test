import easyocr
import cv2
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re

# NLTK 리소스 다운로드
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# EasyOCR 리더 초기화 (영어와 한국어 지원)
reader = easyocr.Reader(['en', 'ko'])

# 이미지 전처리 함수
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# 텍스트 요약 함수
def summarize_text(text):
    sentences = sent_tokenize(text)
    summary = ' '.join(sentences[:2])  # 첫 두 문장 요약
    return summary

# 키워드 추출 함수
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    keywords = word_counts.most_common(5)
    return [word for word, count in keywords]

# 온도 인식 함수
def extract_temperature(text):
    # 정규 표현식을 사용하여 "22°"와 같은 형식에서 숫자와 기호를 정확하게 추출
    match = re.findall(r'(\d+)\s*°', text)
    
    # "°" 없이 숫자만 있는 경우 처리
    if not match:
        match = re.findall(r'(\d+)', text)  # 숫자만 추출
        # 기호가 포함된 다른 텍스트에서 "°"로 추정할 수 있는 패턴을 추가하는 방법
        # 예: "22 체감온도"와 같은 경우
        if match:
            return [int(temp) for temp in match if int(temp) < 100]  # 100 이하의 숫자만 추출

    return [int(temp) for temp in match]

# 메인 함수
def main(image_path):
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
        return

    preprocessed_image = preprocess_image(image)

    # 텍스트 인식
    result = reader.readtext(preprocessed_image)
    extracted_text = ' '.join([text for (_, text, _) in result])

    if not extracted_text:
        print("인식된 텍스트가 없습니다.")
        return

    print(f"추출된 텍스트: {extracted_text}")

    # 텍스트 요약 및 키워드 추출
    summary = summarize_text(extracted_text)
    keywords = extract_keywords(extracted_text)
    temperatures = extract_temperature(extracted_text)

    # 결과 출력
    print(f"요약: {summary}")
    print(f"키워드: {', '.join(keywords)}")
    print(f"인식된 온도: {temperatures}")

# 이미지 파일 경로 설정
image_path = "img\\ticket4.jpg"  # 여기에 이미지 경로를 입력하세요

# 메인 함수 호출
if __name__ == "__main__":
    main(image_path)
