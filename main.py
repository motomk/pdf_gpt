import gradio as gr
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# 質問テンプレート
template = """
あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
質問：{question}
回答：
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)


def process_input(pdf_file, input_text):
    # PDFファイルの読み込み
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    # テキストの分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 埋め込みの作成
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)

    # RetrievalQAの作成
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff",
                                     retriever=vectordb.as_retriever())

    # 質問の送信と回答の取得
    question = input_text
    query = prompt.format(question=question)
    response = qa.run(query)
    source = qa._get_docs(query)[0]
    source_sentence = source.page_content
    answer_source = source_sentence + "\n" + "source:" + source.metadata["source"] + ", page:" + str(
        source.metadata["page"])

    return response + "\n\nSource：\n" + answer_source


# UIコンポーネントの作成
pdf_upload = gr.inputs.File(type="file", label="PDFファイルをアップロード")
textarea = gr.inputs.Textbox(lines=15, placeholder="GPTの応答がここに表示されます...", label="GPT")
input_box = gr.inputs.Textbox(lines=1, placeholder="ここに質問を入力してください", label="")

iface = gr.Interface(
    fn=process_input,
    inputs=[pdf_upload, input_box],
    outputs=textarea,
    layout="vertical",
    css=".gr-input {width: 80%;}",
    allow_flagging='never'
)

iface.launch()
