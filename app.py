# app.py
from flask import Flask, render_template, request, jsonify
import os
import io
import base64
import tempfile
from werkzeug.utils import secure_filename
from preprocessing import process_image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# In-memory store for demo uploads: filename -> bytes
# (Small-scale demo only. Server restart clears it.)
UPLOAD_STORE = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def guess_mime(filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext in ('jpg','jpeg'): return "image/jpeg"
    if ext == 'png': return "image/png"
    if ext in ('tif','tiff'): return "image/tiff"
    if ext == 'bmp': return "image/bmp"
    return "application/octet-stream"

def bytes_to_datauri(b: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode('ascii')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Expects multipart/form-data with key 'file'.
    Returns JSON:
      { success: True, filename: <name>, preview_url: <data-uri> }
    The frontend will then POST { filename } to /process.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'success': False}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'success': False}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type', 'success': False}), 400

    safe_name = secure_filename(file.filename)
    file_bytes = file.read()

    # store in memory under the sanitized name; if same name uploaded twice, newer replaces older.
    UPLOAD_STORE[safe_name] = file_bytes

    mime = guess_mime(safe_name)
    preview_data = bytes_to_datauri(file_bytes, mime=mime)

    return jsonify({
        'success': True,
        'filename': safe_name,
        'preview_url': preview_data
    })

@app.route('/process', methods=['POST'])
def process():
    """
    Expects JSON body: { "filename": "<filename_returned_from_upload>" }
    Uses in-memory upload bytes, writes a temp input file, runs preprocessing into a temp dir,
    reads outputs and returns them as data URIs:
      { success: True, original: <data-uri>, outputs: [ { url: <data-uri>, label: <label>, filename: <fname> }, ... ] }
    """
    try:
        data = request.get_json(force=True)
        filename = data.get('filename')
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'}), 400

        entry_bytes = UPLOAD_STORE.get(filename)
        if entry_bytes is None:
            return jsonify({'success': False, 'error': 'Upload not found or expired'}), 404

        # Use a temporary directory for the preprocessing to write into
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, filename)
            with open(input_path, 'wb') as f:
                f.write(entry_bytes)

            # Call preprocessing (it will write outputs into tmpdir)
            results = process_image(input_path, tmpdir)  # returns list of (filename, label)

            outputs = []
            for out_fname, label in results:
                out_path = os.path.join(tmpdir, out_fname)
                if not os.path.exists(out_path):
                    # skip missing outputs silently
                    continue
                with open(out_path, 'rb') as f:
                    b = f.read()
                datauri = bytes_to_datauri(b, mime="image/png")
                outputs.append({
                    'url': datauri,
                    'label': label,
                    'filename': out_fname
                })

        # Optionally remove the upload from memory to avoid growth. If you want multiple /process calls,
        # comment out the next two lines.
        try:
            del UPLOAD_STORE[filename]
        except KeyError:
            pass

        # original (data-uri)
        orig_mime = guess_mime(filename)
        orig_data = bytes_to_datauri(entry_bytes, mime=orig_mime)

        return jsonify({
            'success': True,
            'original': orig_data,
            'outputs': outputs
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
