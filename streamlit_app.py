import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ABSA Sentiment Analysis", layout="centered")

# กำหนดสไตล์ CSS ให้ดูสะอาดและอ่านง่าย
st.markdown(
    """
    <style>
        .container {
            max-width: 700px;
            margin: auto;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput input {
            border: 2px solid #403c4d;
            border-radius: 5px;
        }
        .content {
            text-align: justify;
            line-height: 1.6;
        }
        .title {
            font-size: 44px;
            font-weight: bold;
            display: flex;
            justify-content: center;     
        }
        .nowrap{
            white-space: nowrap;
        }
        .result {
            display: flex;
            align-items: center;
            gap: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ส่วนหัวของเว็บแอป
st.markdown("<div class='container'>", unsafe_allow_html=True)
st.markdown("<div class='title' style='margin-top: 0;'><span>📌</span><span class='nowrap'>Aspect-based Sentiment Analysis (ABSA)</span></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='content'>
        <h4>📍 วิธีการใช้งานเว็บไซต์</h4>
        <p>
            &emsp;1.เลือกพาดหัวข่าวเกี่ยวกับหุ้นที่สนใจโดยมีเงื่อนไขดังนี้</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นไทย</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นในปี พ.ศ.2566-2567</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นที่มีสัญลักษณ์หุ้นชัดเจน เช่น </br>
            &emsp;&emsp;&emsp;   TISCO ปันผลดี เหมาะสะสม บล.ดีบีเอสฯให้เป้า 118 บ.</br>
            &emsp;2.นำพาดหัวข่าวใส่ลงช่องว่างด้านล่าง</br>
            &emsp;3.กดปุ่ม Apply เพื่อวิเคราะห์
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# กล่องรับข้อความ
user_input = st.text_input("✍️ ใส่ข้อความตรงนี้ :", "")

# ปุ่ม Apply
if st.button("Apply"):
    if user_input:
        st.markdown(
            f"""
            <div class='result'>
                <span style='color: green; font-weight: bold;'>✅ ข้อความที่คุณป้อน :</span>
                <span>{user_input}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ กรุณากรอกข้อความก่อนกด Apply")
    
st.markdown("</div>", unsafe_allow_html=True)
