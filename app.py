import streamlit as st
import os
import sys
import time

# Check if we're in Streamlit Cloud and configure PyTorch accordingly
if 'STREAMLIT_SHARING_MODE' in os.environ:
    os.environ['TORCH_CUDA_VERSION'] = 'cu118'  # Use CUDA 11.8 if available
    os.environ['TORCH_CPU_ONLY'] = '1'  # Force CPU if needed

try:
    import torch
    st.success("✅ PyTorch imported successfully")
except ImportError as e:
    st.error(f"❌ PyTorch import failed: {e}")
    st.info("Trying to install PyTorch...")
    
    # This is a fallback - in Streamlit Cloud, dependencies should be in requirements.txt
    st.stop()

from model1 import TransformerSeq2Seq
from tokenizers import BertWordPieceTokenizer

# Page configuration
st.set_page_config(
    page_title="Urdu Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .urdu-text {
        font-family: 'Noto Sans Arabic', 'Segoe UI', sans-serif;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UrduChatbot:
    def __init__(self, model_path, vocab_path):
        # Use CPU on Streamlit Cloud for compatibility
        self.device = torch.device('cpu')
        st.info(f"Using device: {self.device}")
        
        self.tokenizer = self.load_tokenizer(vocab_path)
        self.model = self.load_model(model_path)
        
        # Get special tokens
        self.CLS_ID = self.tokenizer.token_to_id("[CLS]")
        self.SEP_ID = self.tokenizer.token_to_id("[SEP]")
        self.PAD_ID = self.tokenizer.token_to_id("[PAD]") or 0
        self.UNK_ID = self.tokenizer.token_to_id("[UNK]") or 1
        
        self.max_len = 64

    def load_tokenizer(self, vocab_path):
        """Load the tokenizer from vocabulary.txt"""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
        
        try:
            tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)
            st.success("✅ Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            raise

    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract configuration
            if 'config' in checkpoint:
                config = checkpoint['config']
                vocab_size = checkpoint.get('vocab_size', self.tokenizer.get_vocab_size())
            else:
                # Default configuration
                config = {
                    "d_model": 256,
                    "num_heads": 2,
                    "enc_layers": 2,
                    "dec_layers": 2,
                    "d_ff": 1024,
                    "dropout": 0.1,
                    "max_len": 64
                }
                vocab_size = self.tokenizer.get_vocab_size()
            
            # Initialize model
            model = TransformerSeq2Seq(
                vocab_size=vocab_size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                enc_layers=config["enc_layers"],
                dec_layers=config["dec_layers"],
                d_ff=config["d_ff"],
                dropout=config["dropout"],
                max_len=config["max_len"],
                pad_id=self.PAD_ID
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            st.success("✅ Model loaded successfully")
            return model
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise

    def encode_text(self, text):
        """Encode text to token IDs"""
        try:
            enc = self.tokenizer.encode(text)
            ids = enc.ids
            
            if self.CLS_ID is not None and self.SEP_ID is not None:
                # Add CLS and SEP tokens if available
                if len(ids) > self.max_len - 2:
                    ids = ids[:self.max_len - 2]
                ids = [self.CLS_ID] + ids + [self.SEP_ID]
            else:
                # Just truncate if no special tokens
                ids = ids[:self.max_len]
                
            return ids
        except Exception as e:
            st.error(f"Error encoding text: {str(e)}")
            return []

    def greedy_decode(self, src_ids):
        """Autoregressive generation"""
        try:
            B = src_ids.size(0)
            src_ids = src_ids.to(self.device)
            src_mask = self.model.create_padding_mask(src_ids).to(self.device)
            
            # Start with CLS token or first token
            if self.CLS_ID is not None:
                cur = torch.full((B, 1), self.CLS_ID, dtype=torch.long, device=self.device)
            else:
                cur = torch.full((B, 1), self.UNK_ID, dtype=torch.long, device=self.device)

            for t in range(self.max_len - 1):
                # Create masks
                tgt_mask = self.model.create_look_ahead_mask(cur.size(1)).to(self.device)
                dec_tgt_pad_mask = self.model.create_padding_mask(cur).to(self.device)
                combined_tgt_mask = dec_tgt_pad_mask * tgt_mask
                
                # Generate next token
                with torch.no_grad():
                    logits = self.model(src_ids, cur, src_mask=src_mask, tgt_mask=combined_tgt_mask)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
                cur = torch.cat([cur, next_token], dim=1)
                
                # Stop if SEP token is generated
                if self.SEP_ID is not None and (next_token == self.SEP_ID).all():
                    break
            
            return cur[:, 1:].cpu().tolist()
            
        except Exception as e:
            st.error(f"Error in greedy decode: {str(e)}")
            return [[]]

    def generate_response(self, input_text):
        """Generate response for input text"""
        if not input_text.strip():
            return "Please enter some text."
            
        try:
            # Encode the input
            tokens = self.encode_text(input_text)
            if not tokens:
                return "Error encoding input text."
                
            # Pad to max length
            padded_ids = tokens + [self.PAD_ID] * (self.max_len - len(tokens))
            src = torch.tensor([padded_ids], dtype=torch.long)
            
            # Generate response
            gen_ids = self.greedy_decode(src)
            
            if not gen_ids or not gen_ids[0]:
                return "No response generated."
            
            # Decode the generated tokens
            filt = [int(x) for x in gen_ids[0] if x != self.PAD_ID and x != self.CLS_ID and x != self.SEP_ID]
            if not filt:
                return "Empty response."
                
            response = self.tokenizer.decode(filt, skip_special_tokens=True).strip()
            
            return response if response else "No meaningful response generated."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Urdu Chatbot - Transformer Model</div>', unsafe_allow_html=True)
    
    # Display system info
    st.sidebar.markdown("### System Information")
    st.sidebar.write(f"Python: {sys.version}")
    try:
        st.sidebar.write(f"PyTorch: {torch.__version__}")
        st.sidebar.write(f"CUDA Available: {torch.cuda.is_available()}")
    except:
        st.sidebar.write("PyTorch: Not available")
    
    # Check for required files
    if not os.path.exists("vocabulary.txt"):
        st.error("❌ 'vocabulary.txt' file not found.")
        st.markdown("""
        <div class="error-box">
        <h4>Missing Vocabulary File</h4>
        <p>Please ensure 'vocabulary.txt' is in the same directory as this app.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not os.path.exists("final_model.pt"):
        st.warning("⚠️ Model file 'final_model.pt' not found.")
        st.markdown("""
        <div class="warning-box">
        <h4>Model File Required</h4>
        <p>Please ensure 'final_model.pt' is in the same directory as this app.</p>
        <p>You can upload it here:</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload final_model.pt", type=['pt', 'pth'])
        if uploaded_file is not None:
            with open("final_model.pt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Model file uploaded successfully! Please refresh the page.")
            st.experimental_rerun()
        return

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot uses a Transformer model trained on Urdu text data.
        
        **Features:**
        - Encoder-Decoder Architecture
        - Multi-Head Attention
        - Positional Encoding
        - Urdu Language Support
        """)
        
        st.header("Example Inputs")
        example_inputs = [
            "السلام علیکم",
            "آپ کا نام کیا ہے؟", 
            "موسم کیسا ہے؟",
            "آپ کیسے ہیں؟",
            "شکریہ"
        ]
        
        for example in example_inputs:
            if st.button(example, key=example):
                st.session_state.user_input = example

    # Initialize chatbot
    try:
        chatbot = UrduChatbot("final_model.pt", "vocabulary.txt")
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        st.info("Please check that all required files are present and try again.")
        return

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # User input
        user_input = st.text_area(
            "Type your message in Urdu:",
            value=st.session_state.get('user_input', ''),
            height=100,
            key="user_input",
            help="Enter your message in Urdu language"
        )
        
        # Generate button
        if st.button("Generate Response", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("🤖 Generating response..."):
                    start_time = time.time()
                    response = chatbot.generate_response(user_input)
                    end_time = time.time()
                    
                    # Display response
                    st.markdown("### Response:")
                    st.markdown(f'<div class="response-box"><div class="urdu-text">{response}</div></div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{(end_time - start_time):.2f}s")
                    with col2:
                        st.metric("Input Length", f"{len(user_input)} chars")
                        
                    # Store in chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append((user_input, response))
            else:
                st.warning("⚠️ Please enter a message first.")

    with col2:
        st.subheader("📊 System Info")
        
        st.info(f"""
        **System Specifications:**
        - Device: {chatbot.device}
        - Vocabulary Size: {chatbot.tokenizer.get_vocab_size()}
        - Max Length: {chatbot.max_len}
        - PyTorch: {torch.__version__}
        """)
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Use proper Urdu punctuation
        - Keep sentences clear and concise  
        - The model works best with complete sentences
        - Responses are generated word-by-word
        """)

    # Chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, (input_text, response_text) in enumerate(st.session_state.chat_history[-3:]):
            with st.expander(f"Conversation {i+1}", expanded=False):
                st.markdown(f"**You:** {input_text}")
                st.markdown(f"**Bot:** {response_text}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and PyTorch | "
        "Transformer Model for Urdu Language Processing"
    )

if __name__ == "__main__":
    main()
