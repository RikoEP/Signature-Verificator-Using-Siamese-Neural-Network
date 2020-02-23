import os
from flask import Flask ,render_template, request ,redirect, url_for
from werkzeug.utils import secure_filename
from siameseNetwork import SiameseNetwork
import uuid

UPLOAD_FOLDER = '.\\static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


class Data:
    name = ""
    basePath = '.\\static'
    original=""
    extraction=""
    result=""
    loss=""

    def __init__(self):
        pass

    def set_original(self,original):
        self.original=original


data = Data()
siameseModel = SiameseNetwork()


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def change_name(filename):
    global data
    name = str(uuid.uuid4())+"." + filename.rsplit('.', 1)[1].lower()
    data.name = name
    return name


@app.route('/')
def hello_world():
    global data
    return render_template('index.html')


@app.route('/upload')
def upload():
    global data
    data = Data()
    return render_template('siamese.html', data=data)


@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    global data
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('broken.html',data='No files selected')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, change_name(filename))
            file.save(filepath)
            data.original = filepath.replace(".", "", 1)
            return render_template('siamese.html', data=data)
        return render_template('broken.html',data='Please insert image file')
    else:
        data = Data()
        return redirect(url_for('upload'))


@app.route('/extract', methods = ['GET', 'POST'])
def extract():
    if request.method == 'POST':
        global data
        # data.extraction = data.original
        data.extraction = siameseModel.extract(data)
        return render_template('siamese.html', data=data)
    else:
        data = Data()
        return redirect(url_for('upload'))


@app.route('/verify', methods = ['GET', 'POST'])
def verify():
    if request.method == 'POST':
        global data
        data.result, data.loss = siameseModel.verify(data)
        print(data.loss)
        data.loss = (1 - data.loss)* 100
        return render_template('siamese.html', data=data)
    else:
        data = Data()
        return redirect(url_for('upload'))


@app.route('/add')
def add():
    return render_template('add.html')


@app.route('/adder', methods = ['GET', 'POST'])
def adder():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('broken.html', data='No files selected')
        if file and allowed_file(file.filename):
            # print(request.form['text'])
            name = request.form['text']
            # name="budi"
            path = UPLOAD_FOLDER + "\\signature_data\\"+name
            if os.path.isdir(path):
                pass
            else:
                os.mkdir(path)
            filepath = os.path.join(path, secure_filename(name + ".jpg"))
            file.save(filepath)
            return render_template('broken.html', data='Signature Successfully Added')
        return render_template('broken.html', data='Please insert image file')
    else:
        return redirect(url_for('add'))


if __name__ == '__main__':
    app.run(debug=True)