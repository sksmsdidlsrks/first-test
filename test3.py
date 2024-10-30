import easyocr
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import os

# EasyOCR 리더 초기화 (영어와 한국어 지원)
reader = easyocr.Reader(['ko', 'en'])

# 한국어 불용어 리스트 정의
korean_stopwords = [
    '이', '가', '은', '는', '에', '의', '을', '를', '있다', '하다', '그', '저', '너', '우리', '들',
    '그것', '이것', '저것', '이런', '저런', '어떤', '무슨', '왜', '어디', '언제', '누구', '모두', '각', '모', '또한'
]

def extract_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"{image_path} 이미지를 로드할 수 없습니다.")
        return None

    result = reader.readtext(image)
    extracted_texts = [text for (_, text, _) in result]
    return " ".join(extracted_texts)

def summarize_text(text):
    # 텍스트를 단어로 토큰화하고 불용어 제거
    words = word_tokenize(text)
    stop_words = set(korean_stopwords + ['the', 'and', 'is', 'in', 'to', 'it'])  # 영어 불용어 추가
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    # 가장 빈도가 높은 단어를 선택
    if filtered_words:
        most_common_word, _ = Counter(filtered_words).most_common(1)[0]
        return most_common_word
    return None

def group_similar_images(image_paths):
    texts = []
    for path in image_paths:
        text = extract_text(path)
        if text:
            texts.append(text)

    # CountVectorizer를 사용하여 텍스트 벡터화
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(vectors)

    # 유사한 이미지 그룹화
    groups = []
    visited = set()

    for i in range(len(cosine_sim)):
        if i in visited:
            continue

        similar_images = [image_paths[i]]
        visited.add(i)

        for j in range(len(cosine_sim)):
            if i != j and cosine_sim[i][j] > 0.5:  # 유사도 기준 (0.5 이상)
                similar_images.append(image_paths[j])
                visited.add(j)

        groups.append(similar_images)

    return groups, [summarize_text(text) for text in texts]

def assign_group_names(groups, summaries):
    # 각 그룹에 대한 이름을 지정
    group_names = {}
    for key, images in enumerate(groups):
        group_names[key] = f"그룹 {key + 1} (주요 단어: {summaries[image_paths.index(images[0])]})"
    return group_names

def display_groups(groups, group_names):
    for key, images in enumerate(groups):
        print(f"{group_names[key]}:")
        for img in images:
            print(f"  - {img}")

# 이미지 파일 경로 설정
image_folder = "img"  # 이미지가 저장된 폴더 경로
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# 메인 함수 실행
if __name__ == "__main__":
    groups, summaries = group_similar_images(image_paths)
    group_names = assign_group_names(groups, summaries)
    display_groups(groups, group_names)
