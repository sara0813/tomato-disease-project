import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from class_info import CLASS_INFO


st.set_page_config(
    page_title="Tomato Disease Monitoring System",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Disease Classification")
st.write("토마토 잎 이미지를 업로드하면 병해 분류 결과를 보여주는 시스템입니다.")

uploaded_file = st.file_uploader(
    "토마토 잎 이미지를 업로드하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("업로드한 이미지")
    st.image(image, use_container_width=True)

    st.info("아직 모델 학습 전입니다. 모델 학습 후 예측 기능이 연결됩니다.")

    st.subheader("분류 클래스 목록")

    for class_name, info in CLASS_INFO.items():
        with st.expander(f"{info['name_ko']} ({class_name})"):
            st.write(f"상태: {info['status']}")
            st.write(f"특징: {info['description']}")
            st.write(f"권장 대응: {info['recommendation']}")
else:
    st.warning("이미지를 업로드하면 결과가 표시됩니다.")