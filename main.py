from load_reader import READER_LLM
from ranker import RERANKER
from knowledge import KNOWLEDGE_VECTOR_DATABASE
from prompt import answer_with_rag
from flask import Flask, request, jsonify
# from transformers import Conversation

# question = user_query

answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)

def is_model(llm):
    """Check if the model is loaded and pipeline is a go."""
    return type(llm)

@app.route('/chat/qanda/', methods=['POST'])
def chat_endpoint(question):
    """Chat interface for RAG query. This function is 
    called when a POST request is sent to /chat/qanda/"""
    if not is_model(READER_LLM):
        return jsonify({'error': 'Camus not loaded.'}), 400

    data = request.get_json()
    question = data.get(f'{qanda}_text',")

        return jsonify({answer_with_rag(question)})
    else:
        return jsonify({'error': 'Camus is sleeping.'}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
