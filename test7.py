import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# 1. MobileNet 모델 불러오기 (사전 학습된 가중치 사용)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# 2. 이미지 전처리 함수 정의
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNet 입력 크기
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# 3. 이미지 분류 함수 정의
def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions[0]

# 예시 사용
image_path = 'img\\KakaoTalk_20241029_133622848.png'
predictions = classify_image(image_path)

# 결과 출력
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i+1}: {label} ({score:.2f})")
