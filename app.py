import streamlit as st
import torch

from tokenizer import encode, decode
from model import SmallGPT


# -------------------------
# Page Settings
# -------------------------

st.set_page_config(
    page_title="Small GPT",
    page_icon="🤖"
)

st.title("🤖 Small GPT")
st.write("Small GPT built from scratch using PyTorch")


# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    model = SmallGPT()

    model.load_state_dict(
        torch.load(
            "small_gpt.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# -------------------------
# User Prompt
# -------------------------

prompt = st.text_input(
    "Enter your prompt:",
    placeholder="Example: Machine learning is"
)


max_tokens = st.slider(
    "Maximum tokens",
    20,
    500,
    100
)


# -------------------------
# Generate
# -------------------------

if st.button("Generate"):

    if not prompt:

        st.warning("Please enter a prompt.")

    else:

        try:

            context = torch.tensor(
                [encode(prompt)],
                dtype=torch.long
            )

            with torch.no_grad():

                for _ in range(max_tokens):

                    context_input = context[:, -64:]

                    logits = model(context_input)

                    logits = logits[:, -1, :]

                    probabilities = torch.softmax(
                        logits,
                        dim=-1
                    )

                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1
                    )

                    context = torch.cat(
                        [context, next_token],
                        dim=1
                    )

            generated_text = decode(
                context[0].tolist()
            )

            st.subheader("Generated Text")

            st.write(generated_text)

        except Exception as e:

            st.error(f"Error: {e}")