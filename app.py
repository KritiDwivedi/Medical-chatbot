from flask import Flask,render_template,request,session
from src.helper import download_embeddings
from dotenv import load_dotenv
import os
from src.prompt import*
from langchain_google_genai import ChatGoogleGenerativeAI
from store_collection import chroma_collection

app=Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")


load_dotenv()

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')


# connect to the llm for more refined answer

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

chain= prompt | llm



@app.route("/")
def index():
    return render_template('chat.html')



# when the user clicks the send button
@app.route("/get",methods=["POST"])
def chat():
    question=request.form["msg"]

    # save the history
    history = session.get("chat_history", [])
    history_text = ""

    for chat in history:
        history_text += f"User: {chat['user']}\n"
        history_text += f"Assistant: {chat['assistant']}\n\n"

    # Retrieve top 3 chunks
    results = chroma_collection.query(
        query_texts=[question],
        n_results=3
    )

    # Build context
    context = "\n\n".join(results["documents"][0])
     # Ask Gemini
    response = chain.invoke(
        {
            "context": context,
            "input": question,
            "history": history_text,
        }
    )
    print("Response:",response.content)

    answer = response.content
    # Save current conversation
    history.append(
        {
            "user": question,
            "assistant": answer
        }
    )

    # Keep only the last 5 exchanges
    history = history[-5:]

    session["chat_history"] = history



    return response.content


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


