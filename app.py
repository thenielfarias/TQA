from transformers import pipeline, TapexTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
import os

# Função principal do aplicativo
def main():
    st.set_page_config(page_title="Table Question Answering", page_icon="📉")
    st.header("Ask questions to your data")

    # Carregar as variáveis de ambiente
    load_dotenv(find_dotenv())
    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

    try:
        # Carregar modelos e tokenizadores
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
        TQAmodel = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

        model_id = "google/flan-t5-base"
        filenames = [
            "tokenizer_config.json", "tokenizer.json", "tf_model.h5", "spiece.model", "special_tokens_map.json", "pytorch_model.bin",
            "model.safetensors",  "generation_config.json", "flax_model.msgpack", "config.json"
        ]
        for filename in filenames:
            downloaded_model_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                token=HUGGING_FACE_API_KEY
            )

        text_gen_tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
        text_gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        text_gen_pipeline = pipeline("text2text-generation", model=text_gen_model, tokenizer=text_gen_tokenizer, max_length=1000)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Carregar arquivo .xlsx
    uploaded_file = st.file_uploader("Choose a .xlsx file...", type="xlsx")

    # Input de texto para a pergunta do usuário
    query = st.text_input("Ask it:")
    
    # Processar o arquivo carregado e a pergunta
    if uploaded_file is not None and query:
        try:
            table = pd.read_excel(uploaded_file)

            # Codificar a tabela e a consulta
            encoding = tokenizer(table=table, query=query, return_tensors="pt")
            outputs = TQAmodel.generate(**encoding)
            temp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answer = temp[0]

            # Gerar resposta completa
            template = f"""
            Generate a complete, well-written response based on the following question and answer:
            QUESTION: {query}
            ANSWER: {answer}
            """
            answer_temp = text_gen_pipeline(template)
            answer_final = answer_temp[0]['generated_text']
            
            with st.expander("Generated Answer:"):
                st.write(answer_final)
        except Exception as e:
            st.error(f"Error processing the file or generating the answer: {e}")

if __name__ == '__main__':
    main()
