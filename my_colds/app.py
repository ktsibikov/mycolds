import json
from flask import (
    Flask,
    request,
    abort,
)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    photo_url = request.args.get('photo_url')
    if not photo_url:
        return abort(400)
    return json.dumps(
        {
            'is_plaque': False
        }
    )


if __name__ == '__main__':
    app.run()