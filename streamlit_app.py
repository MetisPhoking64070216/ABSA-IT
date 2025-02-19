import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ABSA Sentiment Analysis", layout="centered")

# กำหนดสไตล์ CSS ให้ดูสะอาดและอ่านง่าย
st.markdown(
    """
    <style>
        .main {
            max-width: 700px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
        }
        .stTextInput input {
            border: 2px solid;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ส่วนหัวของเว็บแอป
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("📌 Aspect-based Sentiment Analysis (ABSA)")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# กล่องรับข้อความ
user_input = st.text_input("✍️ ใส่ข้อความตรงนี้:", "")

# ปุ่ม Apply
if st.button("Apply"):
    if user_input:
        st.success("✅ ข้อความที่คุณป้อน: ")
        st.write(user_input)
    else:
        st.warning("⚠️ กรุณากรอกข้อความก่อนกด Apply")
    
st.markdown("</div>", unsafe_allow_html=True)
