import json
from flask import (
    Flask,
    request,
    abort,
)

from my_colds.detection.sore_throat.sore_throat_detections import PlaqueDetection

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    photo_url = request.args.get('photo_url')
    res = None
    if not photo_url:
        return abort(400)
    try:
        detector = PlaqueDetection(photo_url)
        not_plaque, plaque = detector.detect()
        res = plaque > not_plaque
    except Exception as e:
        abort(500, 'Something went wrong...')

    return json.dumps(
        {
            'is_plaque': res
        }
    )


if __name__ == '__main__':
    app.run()