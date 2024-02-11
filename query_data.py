import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from love import OPENAIKEY
import os
from create_database import createdatabase_main
import warnings


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# Suppress specific deprecation warnings from urllib3 and langchain
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", message="Importing embeddings from langchain is deprecated")
warnings.filterwarnings("ignore", message="Importing chat models from langchain is deprecated")


# Make a new Vector DB
createdatabase_main()
def main():
    # Get input from the user via the terminal.
    prompt = input("Enter your query here: ")

    # Save the query to a txt file in the specified directory with a unique name
    # Extract the first two or three words from the query as the filename
    words = prompt.split()
    file_name = '_'.join(words[:3]) if len(words) >= 3 else '_'.join(words)
    file_path = os.path.join('data/books', file_name)

    # Ensure the filename is unique by appending numbers if necessary
    counter = 1
    unique_file_path = file_path + '.txt'
    while os.path.exists(unique_file_path):
        unique_file_path = f"{file_path}_{counter}.txt"
        counter += 1

    # Write the query to the file
    with open(unique_file_path, 'w') as file:
        file.write(prompt)

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAIKEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(prompt, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_prompt = prompt_template.format(context=context_text, question=prompt)
    #print(chat_prompt)

    model = ChatOpenAI(openai_api_key=OPENAIKEY)
    response_text = model.predict(chat_prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


while True:
    main()
