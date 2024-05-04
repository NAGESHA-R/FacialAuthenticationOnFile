from flask import Flask, render_template, request, redirect, session
import os
import subprocess

app = Flask(__name__)


app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_user', methods=['GET', 'POST'])
def new_user():
    if request.method == 'POST':
        session['username'] = request.form['username']
        username = session['username']
        return f"Your'{username}' created successfully!"
    return render_template('new_user.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        id = request.form['id']
        os.system("python datacollect.py " + id)
        return "Dataset Collection Done.."
    return render_template('capture.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        subprocess.run(["python","trainingdemo.py"])
        return "Training Completed!"
    except Exception as e:
        return str(e), 500

@app.route('/test', methods=['POST'])
def test():
    try:
        username = session.get('username', None)
        if username:
            subprocess.run(["python", "testmodel.py", username])
            return "Model Testing Completed!"
        else:
            subprocess.run(["python", "testmodel.py"])
            return "Model Testing Completed!"
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
