import streamlit as st
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== MODEL ARCHITECTURE ====================
# Copy your entire model architecture here (MultiHeadAttention, PositionalEncoding, 
# EncoderLayer, DecoderLayer, Transformer classes)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attention_scores = None

    def forward(self, q, k, v, mask=None):
        if q.dim() == 2:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        seq_len_q, batch_size, _ = q.shape
        seq_len_k = k.shape[0]
        seq_len_v = v.shape[0]
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        x = x.transpose(0, 1)
        
        if squeeze_output:
            x = x.squeeze(1)
        
        return x

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            seq_len_q = scores.shape[-2]
            seq_len_k = scores.shape[-1]
            if mask.shape[-2] > seq_len_q or mask.shape[-1] > seq_len_k:
                mask = mask[..., :seq_len_q, :seq_len_k]
            scores = scores.masked_fill(mask == 1, -1e9)
        
        p_attn = scores.softmax(dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimension of the embedding.
            h (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor from the previous layer.
            mask (torch.Tensor): The mask for the input sequence.
        
        Returns:
            torch.Tensor: The output tensor of the encoder layer.
        """
        # --- First sub-layer: Multi-Head Self-Attention ---
        # The query, key, and value are all the same: the input 'x'. This is "self-attention".
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask)
        
        # Apply the first residual connection ("Add") and Layer Normalization ("Norm")
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Second sub-layer: Feed-Forward Network ---
        ff_output = self.feed_forward(x)
        
        # Apply the second residual connection and Layer Normalization
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimension of the embedding.
            h (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward network.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input from the previous decoder layer.
            encoder_output (torch.Tensor): The final output of the encoder stack.
            source_mask (torch.Tensor): The mask for the encoder output.
            target_mask (torch.Tensor): The mask for the decoder input.
        
        Returns:
            torch.Tensor: The output tensor of the decoder layer.
        """
        # --- First sub-layer: Masked Multi-Head Self-Attention ---
        attn_output = self.self_attn(q=x, k=x, v=x, mask=target_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Second sub-layer: Encoder-Decoder Cross-Attention ---
        # Query comes from the decoder, Key and Value come from the encoder.
        attn_output = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=source_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # --- Third sub-layer: Feed-Forward Network ---
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_encoder_layers, num_decoder_layers, 
                 num_heads, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        encoder_output = src_emb
        for layer in self.encoder:
            encoder_output = layer(encoder_output, mask=src_mask)
        
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        decoder_output = tgt_emb
        
        tgt_len = tgt.shape[0]
        tgt_causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
        
        for layer in self.decoder:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_causal_mask)
        
        output = self.generator(decoder_output)
        return output


# ==================== UTILITY FUNCTIONS ====================

def normalize_text(text):
    """Normalize text: lowercase, clean whitespace"""
    text = text.lower().strip()
    text = ' '.join(text.split())
    return text


def generate_response_greedy(model, tokenizer, input_text, emotion, situation, max_len=50, device="cpu"):
    """Greedy decoding"""
    normalized_situation = normalize_text(situation)
    normalized_input = normalize_text(input_text)
    prompt = f"Emotion: {emotion} | Situation: {normalized_situation} | Customer: {normalized_input} Agent:"
    
    bos_token_id = tokenizer.token_to_id('<bos>')
    eos_token_id = tokenizer.token_to_id('<eos>')
    
    src_tokens = [bos_token_id] + tokenizer.encode(prompt).ids + [eos_token_id]
    src = torch.tensor(src_tokens).unsqueeze(1).to(device)
    
    tgt_tokens = [bos_token_id]
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt = torch.tensor(tgt_tokens).unsqueeze(1).to(device)
            output = model(src, tgt, src_mask=None)
            last_token_logits = output[-1, 0, :]
            next_token_id = last_token_logits.argmax().item()
            tgt_tokens.append(next_token_id)
            
            if next_token_id == eos_token_id:
                break
    
    generated_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return generated_text


def beam_search_decode(model, tokenizer, input_text, emotion, situation, beam_width=5, max_len=50, device="cpu"):
    """Beam search decoding"""
    normalized_situation = normalize_text(situation)
    normalized_input = normalize_text(input_text)
    prompt = f"Emotion: {emotion} | Situation: {normalized_situation} | Customer: {normalized_input} Agent:"
    
    bos_token_id = tokenizer.token_to_id('<bos>')
    eos_token_id = tokenizer.token_to_id('<eos>')
    
    src_tokens = [bos_token_id] + tokenizer.encode(prompt).ids + [eos_token_id]
    src = torch.tensor(src_tokens).unsqueeze(1).to(device)
    
    model.eval()
    
    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        encoder_output = src_emb
        for layer in model.encoder:
            encoder_output = layer(encoder_output, mask=None)
        
        beams = [(torch.tensor([bos_token_id], device=device), 0.0)]
        
        for _ in range(max_len - 1):
            candidates = []
            all_ended = True
            
            for seq, score in beams:
                if seq[-1].item() == eos_token_id:
                    candidates.append((seq, score))
                    continue
                
                all_ended = False
                tgt_input = seq.unsqueeze(1)
                tgt_emb = model.pos_encoder(model.embedding(tgt_input) * math.sqrt(model.d_model))
                
                decoder_output = tgt_emb
                for layer in model.decoder:
                    tgt_len = decoder_output.shape[0]
                    tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
                    decoder_output = layer(decoder_output, encoder_output, source_mask=None, target_mask=tgt_mask)
                
                logits = model.generator(decoder_output)
                log_probs = F.log_softmax(logits[-1, 0, :], dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_token = top_indices[i].unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=0)
                    new_score = score + top_log_probs[i].item()
                    candidates.append((new_seq, new_score))
            
            if all_ended:
                break
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best_seq, _ = beams[0]
        generated_text = tokenizer.decode(best_seq.tolist(), skip_special_tokens=True)
        return generated_text


# ==================== STREAMLIT APP ====================

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer (cached)"""
    device = torch.device("cpu")  # Use CPU for Streamlit Cloud
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Initialize model with same hyperparameters as training
    vocab_size = tokenizer.get_vocab_size()
    model = Transformer(
        vocab_size=vocab_size,
        d_model=512,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load("best_chatbot_model.pth", map_location=device))
    model.eval()
    
    return model, tokenizer, device


def main():
    st.set_page_config(page_title="Empathetic Chatbot", page_icon="💬", layout="wide")
    
    st.title("💬 Empathetic Conversational Chatbot")
    st.markdown("*Transformer-based chatbot trained on Empathetic Dialogues dataset*")
    
    # Load model
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Emotion selection
    emotions = [
        "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
        "ashamed", "caring", "confident", "content", "devastated", "disappointed",
        "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
        "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
        "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting"
    ]
    
    selected_emotion = st.sidebar.selectbox("Select Emotion", emotions)
    
    # Decoding strategy
    decoding_strategy = st.sidebar.radio(
        "Decoding Strategy",
        ["Greedy", "Beam Search"],
        help="Greedy: Faster, Beam Search: Better quality"
    )
    
    beam_width = 5
    if decoding_strategy == "Beam Search":
        beam_width = st.sidebar.slider("Beam Width", 2, 10, 5)
    
    max_length = st.sidebar.slider("Max Response Length", 20, 100, 50)
    
    # Initialize conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input")
        
        situation = st.text_area(
            "Situation (context)",
            placeholder="Describe the situation...",
            height=100,
            key="situation_input"
        )
        
        user_message = st.text_area(
            "Your Message",
            placeholder="Type your message here...",
            height=100,
            key="user_message_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generate_btn = st.button("🚀 Generate Response", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear History", use_container_width=True)
    
    with col2:
        st.subheader("💭 Response")
        
        if generate_btn:
            if not user_message.strip():
                st.warning("Please enter a message!")
            else:
                with st.spinner("Generating response..."):
                    try:
                        if decoding_strategy == "Greedy":
                            response = generate_response_greedy(
                                model, tokenizer, user_message, 
                                selected_emotion, situation, max_length, device
                            )
                        else:
                            response = beam_search_decode(
                                model, tokenizer, user_message, 
                                selected_emotion, situation, beam_width, max_length, device
                            )
                        
                        # Add to history
                        st.session_state.history.append({
                            'emotion': selected_emotion,
                            'situation': situation,
                            'user': user_message,
                            'bot': response
                        })
                        
                        st.success("Response generated!")
                        st.markdown(f"**Bot:** {response}")
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
        
        if clear_btn:
            st.session_state.history = []
            st.rerun()
    
    # Conversation History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for idx, exchange in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Exchange {len(st.session_state.history) - idx} - Emotion: {exchange['emotion']}", expanded=(idx==0)):
                if exchange['situation']:
                    st.markdown(f"**Situation:** {exchange['situation']}")
                st.markdown(f"**You:** {exchange['user']}")
                st.markdown(f"**Bot:** {exchange['bot']}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with Streamlit | Transformer Model trained from scratch</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
