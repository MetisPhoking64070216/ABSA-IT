import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from huggingface_hub import hf_hub_download

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ABSA Sentiment Analysis", layout="centered")

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
        .content {
            text-align: justify;
            line-height: 1.6;
        }
        .under {
            text-decoration-line: underline;
            text-decoration-style: double;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# โหลดโมเดลจาก Hugging Face
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id="firstmetis/absa_it", filename="model.pth")

        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
        special_tokens = ['<SYMBOL>', '<ASPECT>', '<OPINION>', '<POS>', '<NEG>', '<NEU>']
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        model.eval()
        return model, tokenizer

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
        return None, None

# โหลดโมเดลและ tokenizer
model, tokenizer = load_model()

# ตรวจสอบว่าโหลดสำเร็จหรือไม่
if model is not None and tokenizer is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_text(input_text):
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                num_beams=4,
                do_sample=True,
                temperature=1.2,
                top_k=50,
                top_p=0.95,
                num_return_sequences=4,
                max_length=50,
                return_dict_in_generate=True,  
                output_scores=True  
            )

        sequences = outputs.sequences
        scores = outputs.scores  

        output_texts = [
            tokenizer.decode(seq, skip_special_tokens=False).replace("</s>", "").replace("<pad>", "").strip()
            for seq in sequences
        ]

        confidences = []
        for seq_scores in scores:
            last_token_logits = seq_scores[-1]  
            probs = torch.nn.functional.softmax(last_token_logits, dim=-1)  
            confidence = probs.max().item()  
            confidences.append(confidence)

        return list(zip(output_texts, confidences))

    # ส่วนหัวของเว็บแอป
    st.title("📌 Aspect-based Sentiment Analysis (ABSA)")
    st.markdown(
    """
    <div class='content'>
        <h4>📍 วิธีการใช้งานเว็บไซต์</h4>
        <p>
            &emsp;1. เลือกพาดหัวข่าวเกี่ยวกับหุ้นที่สนใจโดยมีเงื่อนไขดังนี้</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นไทยในปี พ.ศ.2566-2567</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นไทยที่มีสัญลักษณ์หุ้นชัดเจน</br>
            &emsp;&emsp;&emsp;- เป็นข่าวหุ้นไทยที่มีการออกข่าวค่อนข้างบ่อย</br>
            &emsp;&emsp;&emsp;   <u class="under">ตัวอย่าง</u> : TISCO ปันผลดี เหมาะสะสม บล.ดีบีเอสฯให้เป้า 118 บ.</br>
            &emsp;2. นำพาดหัวข่าวใส่ลงช่องว่างด้านล่าง</br>
            &emsp;3. กดปุ่ม Apply เพื่อวิเคราะห์
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
    st.markdown("ใส่พาดหัวข่าวหุ้น เพื่อวิเคราะห์ Sentiment")

    # กล่องรับข้อความ
    user_input = st.text_input("✍️ ใส่ข้อความตรงนี้ :", "")

    # ปุ่ม Apply
    if st.button("Apply"):
        if user_input:
            results = generate_text(user_input)

            # 🔹 แสดงผลลัพธ์ทั้งหมด พร้อมเปอร์เซ็นต์ความมั่นใจ
            st.markdown("### 🔍 ผลลัพธ์:")
            for i, (text, confidence) in enumerate(results, 1):
                st.markdown(f"**🔹 ผลลัพธ์ {i}:** {text}  \n📌 **ความมั่นใจ:** {confidence:.2%}")

        else:
            st.warning("⚠️ กรุณากรอกข้อความก่อนกด Apply")
