import os, base64, logging, json
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI  # 1.x SDK

# ── Streamlit 기본 설정 ───────────────────────────
st.set_page_config(page_title="헬멧 판별기", layout="centered")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ── OpenAI 초기화 ─────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 화면 구성 ────────────────────────────────────
st.title("🏍️ 헬멧 착용 여부 판별기")
st.write("사진을 업로드하면 GPT-4o Vision으로 헬멧 착용 여부를 알려줍니다.")

uploaded_file = st.file_uploader("📸 사진 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="업로드된 이미지", use_column_width=True)

    if st.button("판별 시작"):
        with st.spinner("판별 중… 잠시만 기다려 주세요"):
            # ── ① 이미지 → data-URL ─────────────────
            buf = BytesIO()
            img.save(buf, format="PNG")
            data_url = (
                "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            )

            # ── ② GPT-4o Vision 호출 ─────────────────
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},  # ← 콤마 추가
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {
                                "type": "text",
                                "text": (
                                    "이 사진에서 사람이 헬멧을 쓰고 있는지 여부를 "
                                    'JSON 형식 {"helmet": bool, "confidence": int} 으로만 답해주세요.'
                                ),
                            },
                        ],
                    }
                ],
            )

            # ── ③ 결과 파싱 및 출력 ───────────────────
            try:
                result = json.loads(resp.choices[0].message.content)
                if result["helmet"]:
                    st.success(f"✅ 헬멧 착용 (신뢰도: {result['confidence']}%)")
                else:
                    st.error(f"❌ 헬멧 미착용 (신뢰도: {result['confidence']}%)")
            except Exception as e:
                st.error(f"GPT 응답 파싱 오류: {e}")
