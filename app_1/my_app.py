from openai import OpenAI
import streamlit as st

st.title("🐍Python Code Debugger")
bg="""
<style>
[data-testid="stAppViewContainer"]{
  background-color:	#4A646C;
}
</style>
"""
st.markdown(bg,unsafe_allow_html=True)

#Read the private key and set up a client
f=open("Keys\.openai_demo_key.txt")
key=f.read()
client=OpenAI(api_key=key)

#Create a input box
label="Enter your python code"
prompt=st.text_area(label)


#If the button is clicked, generate the output
if st.button("Generate")==True:
  response= client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
      {'role':'system','content':"""You are a helpful AI assistant.
            Given a python code show where the errors or bugs are present and also debug and give the correct code."""},
      {'role':'user','content':prompt}
      ]
  )
  st.write(response.choices[0].message.content)

