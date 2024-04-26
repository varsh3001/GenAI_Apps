from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

def rag_system(query):

  chat_model = ChatGoogleGenerativeAI(google_api_key="google_api_key.txt", model="gemini-1.5-pro-latest")

  loader = PyPDFLoader("leave_paper.pdf")
  pages = loader.load_and_split()

  text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = text_splitter.split_documents(pages)

  embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyCrXQdAoDMaRaDL6pL5BcgiEZj9HPR5uP4", model="models/embedding-001")
  db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# Persist the database on drive
  db.persist()
  db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
  retriever = db_connection.as_retriever(search_kwargs={"k": 15})
  retrieved_docs = retriever.invoke(query)
  output_parser = StrOutputParser()
  chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Assistant named Pippo.
    You analyze the context and question from the user. Your answer should be based on the specific context."""),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)


  response=rag_chain.invoke(query)
  return response
