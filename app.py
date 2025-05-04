from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

app = Flask(__name__)


df = pd.read_csv("shl_assessments.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['description'] = df['assessment_name'].fillna('') + ' ' + df['test_types'].fillna('')


model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['description'].apply(lambda x: model.encode(x))

@app.route('/')
def home():
    with open("index.html", "r") as f:
        return f.read()

@app.route('/api/recommend', methods=['POST'])
def recommend():
    desc = request.json.get('job_description', '')
    if not desc.strip():
        return jsonify({'result': 'Please provide a valid job description.'})

    query_embedding = model.encode(desc)
    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity([query_embedding], [emb])[0][0])
    top_matches = df.sort_values('similarity', ascending=False).head(10)

    if top_matches.empty:
        return jsonify({'result': 'No relevant assessments found.'})

    html = "<table><tr><th>Assessment</th><th>URL</th><th>Remote</th><th>Adaptive</th><th>Test Type</th></tr>"
    for _, row in top_matches.iterrows():
        html += f"<tr><td>{row['assessment_name']}</td><td><a href='{row['url']}' target='_blank'>Link</a></td><td>{row.get('remote_testing_support', 'N/A')}</td><td>{row.get('adaptive/irt_support', 'N/A')}</td><td>{row.get('test_types', 'N/A')}</td></tr>"
    html += "</table>"

    return jsonify({'result': html})


@app.route('/api/json_recommend', methods=['POST'])
def json_recommend():
    desc = request.json.get('job_description', '')
    if not desc.strip():
        return jsonify({'error': 'Please provide a valid job description.'}), 400

    query_embedding = model.encode(desc)
    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity([query_embedding], [emb])[0][0])
    top_matches = df.sort_values('similarity', ascending=False).head(5)

    if top_matches.empty:
        return jsonify({'recommendations': []})

    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "assessment_name": row['assessment_name'],
            "url": row['url'],
            "remote_testing_support": row.get('remote_testing_support', 'N/A'),
            "adaptive_support": row.get('adaptive/irt_support', 'N/A'),
            "test_type": row.get('test_types', 'N/A'),
            "similarity": round(row['similarity'], 3)
        })

    return jsonify({'recommendations': results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)