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
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stTextInput input {
            border: 2px solid #4CAF50;
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
            gap: 10px;
        }
        span{
            white-space: nowrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ส่วนหัวของเว็บแอป
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'><span>📌</span><span>Aspect-based Sentiment Analysis (ABSA)</span></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='content'>
        <h4>ที่มาและความสำคัญ</h4>
        <p>
        &emsp;&emsp;ABSA เกิดขึ้นจากความต้องการที่จะเข้าใจความคิดเห็นของผู้ใช้ในเชิงลึก เพราะการวิเคราะห์ความรู้สึก</br>
        โดยทั่วไปอาจบ่งบอกได้แค่ความคิดเห็นแบบโดยรวม แต่ ABSA มุ่งเน้นไปที่การเข้าใจความคิดเห็นในแต่ละด้าน</br>
        เช่น คุณภาพของสินค้าหรือการบริการลูกค้า ซึ่งช่วยให้เข้าใจความคิดเห็นของลูกค้าในแต่ละด้านของผลิตภัณฑ์ได้ชัดเจนขึ้น
        </p>
        <p>
        &emsp;&emsp;ด้วยปริมาณที่เพิ่มขึ้นของข้อมูลดิจิทัลในปัจจุบัน ทำให้ ABSA ถูกนำมาใช้งานมากยิ่งขึ้น ในโครงงานนี้
        จึงได้นำ ABSA มาใช้ในการวิเคราะห์ข่าวหุ้น ว่าข่าวดังกล่าวมีหุ้นอะไรที่ถูกพูดถึงบ้าง และมีผลเชิงบวก
        หรือเชิงลบอย่างไรกับหุ้นดังกล่าว นักลงทุนมีความคิดเห็นอย่างไรกับหุ้น ซึ่งสิ่งเหล่านี้สามารถทำให้ผู้ที่ไม่มี
        ความรู้เกี่ยวกับหุ้นก็สามารถเข้าใจถึงสภาพของหุ้นนั้นๆ ได้
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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
