import streamlit as st
import requests
from PIL import Image
import io
import subprocess
import sys
import time
import os

st.set_page_config(
    page_title="HealthTrust - AI Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
ML_API_URL = "http://localhost:8000"

# Auto-start ML Service (for Streamlit Cloud)
@st.cache_resource
def start_ml_service():
    # Check if we are running on Streamlit Cloud (or just want to ensure it runs)
    # We will try to start it if it feels like it's not running, or just run it blindly
    # since uvicorn inside main.py usually handles port binding.
    
    # Path to the ml-service main.py
    # Assuming app.py is in streamlit-app/ and main.py is in ml-service/
    ml_service_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml-service", "main.py"))
    
    if not os.path.exists(ml_service_path):
        st.error(f"Cannot find ML service at {ml_service_path}")
        return None

    # Get API key from secrets (Cloud) or environment (Local)
    api_key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except FileNotFoundError:
        pass  # No secrets.toml found, likely running locally without it

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    # Prepare environment variables for the subprocess
    env = os.environ.copy()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    
    # Start the process
    # We use sys.executable to ensure we use the same python interpreter (with installed deps)
    process = subprocess.Popen(
        [sys.executable, ml_service_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(ml_service_path), # Set CWD to ml-service dir
        env=env # Pass environment variables including API key
    )
    
    # Give it a moment to start
    time.sleep(10) # Increased wait time to 10s

    
    # Check if it died immediately
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        st.error(f"❌ ML Service Failed to Start!")
        st.code(f"Exit Code: {process.returncode}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
        return None

    return process


# Initialize service
ml_process = start_ml_service()

# Subprocess Health Check & Debugging
if 'ml_process' in locals() and ml_process:
    if ml_process.poll() is not None:
        st.sidebar.error("❌ Process Crashed")
        stdout, stderr = ml_process.communicate()
        st.sidebar.error(f"Exit: {ml_process.returncode}")
        if stderr:
            st.sidebar.code(stderr[-200:]) # Show last 200 chars
    else:
        st.sidebar.success(f"✅ PID: {ml_process.pid} (Running)")
else:
    st.sidebar.warning("⚠️ No ML Process Tracked")

# Debugging Button
if st.sidebar.button("Show Process Logs"):
    if 'ml_process' in locals() and ml_process:
        st.sidebar.text("Reading logs...")
        # Note: communicate() would block, so we can't easily read continuously without threads.
        # But if it's dead, we can read.
        if ml_process.poll() is not None:
             o, e = ml_process.communicate()
             st.sidebar.text(o)
             st.sidebar.text(e)
        else:
             st.sidebar.info("Process is alive. Logs accessible only if it stops.")






# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏥 HealthTrust Insurance</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Fraud Detection System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏥 HealthTrust Insurance")
st.sidebar.markdown("### AI-Powered Fraud Detection")
st.sidebar.markdown("---")

# Add API status check
try:
    response = requests.get(f"{ML_API_URL}/", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ ML Service Online")
    else:
        st.sidebar.error("⚠️ ML Service Error")
except:
    st.sidebar.error("❌ ML Service Offline")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Both services must be running for full functionality.")


# ============================================================================
# SUBMIT CLAIM PAGE - INTEGRATED WORKFLOW
# ============================================================================
st.markdown("## 📋 Submit Insurance Claim")
st.markdown("Fill in the claim details and upload images to get comprehensive fraud detection analysis.")


with st.form("claim_form"):
    st.markdown("### 📝 Claim Details")
    col1, col2 = st.columns(2)
    
    with col1:
        amount_billed = st.number_input(
            "💰 Amount Billed (ADA)",
            min_value=0.0,
            max_value=1000000.0,
            value=500.0,
            step=50.0,
            help="Total amount claimed in ADA"
        )
        
        age = st.number_input(
            "👤 Patient Age",
            min_value=1,
            max_value=120,
            value=35,
            step=1,
            help="Age of the patient"
        )
    
    with col2:
        gender = st.selectbox(
            "⚧ Gender",
            options=["Male", "Female"],
            help="Patient's gender"
        )
        
        diagnosis = st.selectbox(
            "🏥 Diagnosis",
            options=[
                'Pregnancy',
                'Hypertension',
                'Diabetes',
                'Pneumonia',
                'Gastroenteritis',
                'Cesarean Section',
                'Cataract Surgery',
                'Other'
            ],
            help="Medical diagnosis category"
        )
    
    st.markdown("---")
    st.markdown("### 📸 Supporting Documents (Optional but Recommended)")
    st.info("💡 **Tip:** Upload both images for the most accurate fraud detection results!")
    
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("**📄 Prescription Image**")
        prescription_file = st.file_uploader(
            "Upload prescription",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the prescription",
            key="prescription"
        )
    
    with img_col2:
        st.markdown("**🧾 Receipt Image**")
        receipt_file = st.file_uploader(
            "Upload receipt",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the receipt",
            key="receipt"
        )
    
    submit_button = st.form_submit_button("🔍 Analyze Claim")

    # Show image previews outside the form
if prescription_file or receipt_file:
    st.markdown("#### 👁️ Image Previews")
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        if prescription_file:
            image = Image.open(prescription_file)
            st.image(image, caption="Prescription Preview", use_column_width=True)
    
    with preview_col2:
        if receipt_file:
            image = Image.open(receipt_file)
            st.image(image, caption="Receipt Preview", use_column_width=True)

if submit_button:
    if not prescription_file or not receipt_file:
        st.error("⚠️ Please upload both prescription and receipt images to proceed with the analysis.")
        st.stop()

    st.markdown("---")
    
        # STEP 1: ML Prediction
    with st.spinner("🤖 Step 1/2: Running ML fraud detection model..."):
        try:
            response = requests.post(
                f"{ML_API_URL}/predict",
                json={
                    "amount_billed": amount_billed,
                    "age": age,
                    "gender": gender,
                    "diagnosis": diagnosis
                },
                timeout=10
            )
            
            if response.status_code != 200:
                st.error(f"❌ ML Prediction Error: {response.text}")
                st.stop()
            
            ml_data = response.json()
            ml_confidence = ml_data["confidence"]
            ml_label = ml_data["prediction_label"]
            
            st.success("✅ ML analysis complete!")
            
        except requests.exceptions.ConnectionError:
            st.error("❌ **Cannot connect to ML Service**")
            st.warning("Make sure the ML service is running")
            st.stop()
        except Exception as e:
            st.error(f"❌ ML Error: {str(e)}")
            st.stop()
    
        # STEP 2: Image Verification (if images provided)
    image_data = None
    if prescription_file and receipt_file:
        with st.spinner("📸 Step 2/2: Verifying prescription and receipt images with GPT-4 Vision..."):
            try:
                    # Reset file pointers
                prescription_file.seek(0)
                receipt_file.seek(0)
                
                files = {
                    "prescription_image": ("prescription.jpg", prescription_file, "image/jpeg"),
                    "receipt_image": ("receipt.jpg", receipt_file, "image/jpeg")
                }
                
                data = {"ml_confidence": ml_confidence}
                
                response = requests.post(
                    f"{ML_API_URL}/verify-images",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    image_data = response.json()
                    st.success("✅ Image verification complete!")
                else:
                    st.warning(f"⚠️ Image verification failed: {response.text}")
                    st.info("Continuing with ML prediction only...")
                    
            except Exception as e:
                st.warning(f"⚠️ Image verification error: {str(e)}")
                st.info("Continuing with ML prediction only...")
    else:
        if prescription_file or receipt_file:
            st.info("ℹ️ Both prescription AND receipt images are required for verification. Showing ML prediction only.")
    
        # DISPLAY RESULTS
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
        # Determine final verdict
    if image_data and 'combined_score' in image_data:
            # Combined scoring available
        combined_score = image_data['combined_score']
        final_status = image_data['final_status']
        
        if final_status == 'genuine':
            st.success(f"### ✅ CLAIM APPROVED")

        else:
            st.error(f"### ❌ CLAIM REJECTED")
        
            # Display all scores
        st.markdown("#### 🎯 Comprehensive Analysis")
        score_col1, score_col2, score_col3 = st.columns(3)
        
        with score_col1:
            ml_score_pct = ml_confidence * 100
            st.metric(
                "🤖 ML Score",
                f"{ml_score_pct:.2f}%",
                help="Machine Learning fraud detection confidence"
            )
        
        with score_col2:
            image_score = image_data['image_verification']['score']
            st.metric(
                "📸 Image Score",
                f"{image_score:.2f}%",
                help="Prescription-Receipt match percentage"
            )
        
        with score_col3:
            st.metric(
                "⭐ Combined Score",
                f"{combined_score:.2f}%",
                delta="APPROVED" if combined_score >= 80 else "REJECTED",
                help="Average of ML and Image scores (≥80% = Approved)"
            )
        
            # Explanation
        st.info(f"""
        **Final Verdict Explanation:**
        - ML Model predicted: **{ml_label.upper()}** with {ml_score_pct:.1f}% confidence
        - Image verification scored: **{image_score:.1f}%** match
        - Combined average: **{combined_score:.1f}%** 
        - Threshold: **80%** (scores ≥80% are approved)
        - **Result: {'✅ APPROVED' if combined_score >= 80 else '❌ REJECTED'}**
        """)
        
    else:
            # ML prediction only
        if ml_label == "genuine":
            st.success(f"### ✅ CLAIM APPROVED (ML Analysis)")
        else:
            st.error(f"### ❌ CLAIM REJECTED (ML Analysis)")
        
        st.warning("⚠️ **Note:** Only ML prediction available. Upload both prescription and receipt images for comprehensive analysis.")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric(
                "Prediction",
                ml_label.upper(),
                delta="Approved" if ml_label == "genuine" else "Rejected"
            )
        
        with detail_col2:
            ml_confidence_pct = ml_confidence * 100
            st.metric(
                "ML Confidence",
                f"{ml_confidence_pct:.2f}%"
            )
        
        with detail_col3:
            st.metric(
                "Amount",
                f"₳{amount_billed:,.2f}"
            )
        
        st.info(f"**{ml_data['message']}**")
    
        # Detailed results expander
    with st.expander("📋 View Technical Details"):
        st.markdown("**ML Prediction Data:**")
        st.json(ml_data)
        if image_data:
            st.markdown("**Image Verification Data:**")
            st.json(image_data)


# ============================================================================
# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🏥 HealthTrust Insurance - AI-Powered Fraud Detection | Built with Streamlit & FastAPI</p>",
    unsafe_allow_html=True
)
