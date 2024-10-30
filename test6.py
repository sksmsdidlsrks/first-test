import easyocr
from collections import Counter

def extract_text(image_path):
    # EasyOCR 리더 생성
    reader = easyocr.Reader(['ko', 'en'])
    
    # 텍스트 추출
    results = reader.readtext(image_path)
    text = " ".join([result[1] for result in results])
    
    return text

def summarize_text(text):
    # 단어 분리 및 소문자 변환
    words = text.lower().split()
    
    # 단어 빈도수 계산
    word_counts = Counter(words)
    
    # 가장 빈번한 단어 찾기
    most_common_word = word_counts.most_common(1)
    
    if most_common_word:
        return most_common_word[0][0]  # 가장 빈번한 단어 반환
    return None

# 사용 예
image_path = 'img\KakaoTalk_20241029_133622848_01.png'  # 실제 이미지 경로로 변경
extracted_text = extract_text(image_path)
summary = summarize_text(extracted_text)

print(f"요약된 단어: {summary}")