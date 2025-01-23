from flask import Flask, request, send_file, render_template, jsonify
from flask_restful import Resource, Api
from pycore.main import Rundown,Rundown_Save

app = Flask(__name__)
api = Api(app)

class NameResource(Resource):
    def post(self):
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')

        # Rufe Klassifikation Rundown(name, vorname) auf
        result,insert = Rundown(last_name, first_name)
        
        # Gib das Ergebnis als JSON zurück, inklusive Erfolgswert
        response_data = {
            'message': result,
            'success': insert
        }

        return jsonify(response_data)

class SaveNameResource(Resource):
    def post(self):
        data = request.get_json()  # Hole die Daten aus dem JSON-Request

        name = data.get('name')
        vorname = data.get('vorname')

        # Rufe deine Funktion Rundown_Save(name, vorname) auf
        result = Rundown_Save(name, vorname)

        # Gib das Ergebnis als JSON zurück, inklusive Erfolgswert
        response_data = {
            'message': result,
            'success': True
        }

        return jsonify(response_data)

api.add_resource(NameResource, '/add_name')
api.add_resource(SaveNameResource, '/save_name')

@app.route('/')
def main_page():
    return send_file('Main.html')

if __name__ == '__main__':
    from waitress import serve
    serve(app,host="127.0.0.1",port=5000)