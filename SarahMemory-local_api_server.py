from flask import Flask, request, jsonify
from flask_cors import CORS
# Import existing SarahMemory AI modules here

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    # Process with existing AI logic
    response = "Your AI response here"
    return jsonify({'response': response})

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'connected', 'online': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)