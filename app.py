import streamlit as st
import torch
import os
from model import TransformerSeq2Seq
from tokenizers import BertWordPieceTokenizer
import time

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
    .input-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class UrduChatbot:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.model = self.load_model(model_path)
        self.CLS_ID = self.tokenizer.token_to_id("[CLS]") if self.tokenizer.token_to_id("[CLS]") is not None else None
        self.SEP_ID = self.tokenizer.token_to_id("[SEP]") if self.tokenizer.token_to_id("[SEP]") is not None else None
        self.PAD_ID = self.tokenizer.token_to_id("[PAD]") if self.tokenizer.token_to_id("[PAD]") is not None else 0
        self.max_len = 64

    def load_tokenizer(self, tokenizer_path):
        """Load the tokenizer"""
        vocab_path = None
        for d in [tokenizer_path, os.path.join(tokenizer_path, "")]:
            cand = os.path.join(d, "vocab.txt")
            if os.path.exists(cand):
                vocab_path = cand
                break
        
        if vocab_path is None:
            raise FileNotFoundError("Could not find vocab.txt in tokenization directories")
        
        return BertWordPieceTokenizer(vocab_path, lowercase=False)

    def load_model(self, model_path):
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint['config']
        vocab_size = checkpoint['vocab_size']
        
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
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def encode_text(self, text):
        """Encode text to token IDs"""
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        if self.CLS_ID is not None and self.SEP_ID is not None:
            ids = [self.CLS_ID] + ids[: self.max_len - 2] + [self.SEP_ID]
        else:
            ids = ids[: self.max_len]
        return ids

    def greedy_decode(self, src_ids):
        """Autoregressive generation"""
        B = src_ids.size(0)
        src_ids = src_ids.to(self.device)
        src_mask = self.model.create_padding_mask(src_ids).to(self.device)
        
        if self.CLS_ID is not None:
            cur = torch.full((B,1), self.CLS_ID, dtype=torch.long, device=self.device)
        else:
            cur = torch.full((B,1), self.PAD_ID, dtype=torch.long, device=self.device)

        for t in range(self.max_len-1):
            tgt_mask = self.model.create_look_ahead_mask(cur.size(1)).to(self.device)
            dec_tgt_pad_mask = self.model.create_padding_mask(cur)
            combined_tgt_mask = dec_tgt_pad_mask * tgt_mask
            
            with torch.no_grad():
                logits = self.model(src_ids, cur, src_mask=src_mask, tgt_mask=combined_tgt_mask)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            cur = torch.cat([cur, next_token], dim=1)
            
            if self.SEP_ID is not None:
                if (next_token == self.SEP_ID).all():
                    break
        
        return cur[:, 1:].cpu().tolist()

    def generate_response(self, input_text):
        """Generate response for input text"""
        # Encode the input
        tokens = self.encode_text(input_text)
        padded_ids = tokens + [self.PAD_ID] * (self.max_len - len(tokens))
        src = torch.tensor([padded_ids], dtype=torch.long)
        
        # Generate response
        gen_ids = self.greedy_decode(src)
        
        # Decode the generated tokens
        filt = [int(x) for x in gen_ids[0] if x != self.PAD_ID and x != self.CLS_ID and x != self.SEP_ID]
        response = self.tokenizer.decode(filt, skip_special_tokens=True).strip()
        
        return response

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Urdu Chatbot - Transformer Model</div>', unsafe_allow_html=True)
    
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
        
        **How to use:**
        1. Type your message in Urdu
        2. Click 'Generate Response'
        3. View the AI-generated response
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
    @st.cache_resource
    def load_chatbot():
        model_path = "best_model.pth"  # Update this path if needed
        tokenizer_path = "tokenization"  # Update this path if needed
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer directory not found at: {tokenizer_path}")
            return None
            
        try:
            chatbot = UrduChatbot(model_path, tokenizer_path)
            return chatbot
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    chatbot = load_chatbot()
    
    if chatbot is None:
        st.error("Failed to load the chatbot. Please check the model files.")
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
                with st.spinner("Generating response..."):
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
            else:
                st.warning("Please enter a message first.")

    with col2:
        st.subheader("📊 Model Info")
        
        if hasattr(chatbot, 'model'):
            st.info(f"""
            **Model Specifications:**
            - Vocabulary Size: {chatbot.model.token_embed.num_embeddings:,}
            - Embedding Dim: {chatbot.model.d_model}
            - Encoder Layers: {len(chatbot.model.enc_layers)}
            - Decoder Layers: {len(chatbot.model.dec_layers)}
            - Attention Heads: {chatbot.model.enc_layers[0].mha.num_heads if chatbot.model.enc_layers else 'N/A'}
            - Device: {chatbot.device}
            """)
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Use proper Urdu punctuation
        - Keep sentences clear and concise
        - The model works best with complete sentences
        - Responses are generated word-by-word
        """)

    # Chat history (optional)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, (input_text, response_text) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
            with st.expander(f"Conversation {i+1}"):
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