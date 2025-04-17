import streamlit as st
from agent import run_agent
import speech_recognition as sr
import io
import soundfile as sf

# Set the page configuration
st.set_page_config(page_title="CrewAI Sentiment Analysis", layout="centered")
st.markdown("<h1 style='text-align: center;'>HAI - HeartAI</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size:16px;'> Delivering AI-Powered Response that feel personal, kind, and impactful</h2>", unsafe_allow_html=True)
st.markdown("Upload an audio file or enter your review below for sentiment analysis:")

# Function to transcribe audio from uploaded file
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        # read audio file using soundfile
        audio_data, sample_rate = sf.read(audio_file)
        # Convert to WAV format for speech recognition
        with io.BytesIO() as wav_io:
            sf.write(wav_io, audio_data, sample_rate, format="WAV")
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"
    except Exception as e:
        return f"Error processing audio: {e}"

# sidebar for input field
def render_sidebar():
    inputs = {}
    with st.sidebar:
        st.title("Review Details")
        inputs["cust_name"] = st.text_input("Enter Customer Name")
        inputs["purch_date"] = st.date_input("Enter Purchase Date", value=None, min_value=None, max_value=None, format="YYYY-MM-DD")
        products = ['Apparel', 'Automotive', 'Baby', 'Beauty', 'Books', 'Camera',
                    'Digital_Ebook_Purchase', 'Digital_Music_Purchase',
                    'Digital_Software', 'Digital_Video_Download',
                    'Digital_Video_Games', 'Electronics', 'Furniture', 'Gift Card',
                    'Grocery', 'Health & Personal Care', 'Major Appliances',
                    'Mobile_Apps', 'Mobile_Electronics', 'Musical Instruments',
                    'Music', 'Office Products', 'Outdoors', 'PC',
                    'Personal_Care_Appliances', 'Pet Products', 'Shoes', 'Software',
                    'Sports', 'Tools', 'Toys', 'Video DVD', 'Video Games', 'Video',
                    'Watches', 'Wireless']
        inputs["product"] = st.selectbox("Select Product Purchased", products)
    return inputs

# Main app section
inputs = render_sidebar()

# create session state for user review
if "user_review" not in st.session_state:
    st.session_state.user_review = ""

# Option to choose input method
input_method = st.radio("Choose Input Method", ["Text", "Upload Audio"])

if input_method == "Text":
    st.session_state.user_review = st.text_area("Your Review", st.session_state.user_review, height=150)
else:
    st.markdown("Upload Audio Review")
    audio_file = st.file_uploader("Choose an audio file (WAV, MP3)", type=["wav", "mp3"])
    if audio_file is not None:
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                transcribed_text = transcribe_audio(audio_file)
                st.session_state.user_review = transcribed_text
                st.write(f"**Transcribed Review:** {transcribed_text}")
    else:
        st.info("Please upload an audio file to transcribe.")

    st.session_state.user_review = st.text_area("Edit Transcribed Review", st.session_state.user_review, height=150)

# button to start analysis
if st.button("Analyze"):
    cust_name = inputs.get("cust_name")
    purch_date = inputs.get("purch_date")
    product = inputs.get("product")
    user_review = st.session_state.user_review
    if not cust_name or not purch_date or not product or len(user_review.split()) < 1:
        st.error("Please fill in all fields (Customer Name, Purchase Date, Product, and at least 1 word in the review).")
    else:
        # Create dictionary input for agent
        agent_input = {
            "cust_name": cust_name,
            "purch_date": str(purch_date),  # Convert date to string for consistency
            "product": product,
            "review": user_review
        }
        with st.spinner("Processing sentiment analysis for the review..."):
            result = run_agent(agent_input)
        
        st.success("Analysis complete!")
        
        st.markdown("### Final Reviewed Response")
        st.markdown(result['reviewed_response'])
        
        with st.expander("View Additional Details", expanded=False):
            st.markdown("**Sentiment Analysis:**")
            st.write(result['sentiment'])
            st.markdown("**Sentiment Review:**")
            st.write(result['sentiment_review'])
            st.markdown("**Generated Response:**")
            st.write(result['response'])
            st.markdown("**Models used:**")
            st.write(result['Used_Model'])
            