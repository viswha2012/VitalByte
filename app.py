from flask import Flask, render_template, request, url_for, redirect
import numpy as np
from keras.models import load_model
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_login import logout_user

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:4c8NZiC9bPPBf0p26yvb@containers-us-west-129.railway.app:7965/railway'
db = SQLAlchemy(app)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Details(db.Model):
    __tablename__ = 'details'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    fullname = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"Details(username='{self.username}', fullname='{self.fullname}')"

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['userid']
    password = request.form['pass']
    fullname = request.form['fullname']

    # Check if the username already exists
    if Details.query.filter_by(username=username).first():
        return 'User already exists!'

    # Create a new Details instance
    new_details = Details(username=username, password=password, fullname=fullname)
    db.session.add(new_details)
    db.session.commit()

    return redirect('/')

@app.route('/login', methods=['POST'])
def auth():
    username = request.form['username']
    password = request.form['password']

    details = Details.query.filter_by(username=username).first()
    if details:
        if details.password == password:
            return redirect('/survey')
    
    return 'Invalid username or password. Please try again.'

# Load the trained model
model = load_model('obesity_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the column names for one-hot encoding
categorical_columns = ['gender', 'family_history_with_overweight', 'caloric_food',
                       'smoke', 'calories', 'transportation']


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        try:
            # Get the form data
            name = request.form['name']
            age = int(request.form['age'])
            male = int(request.form['gender'] == 'male')
            female = int(request.form['gender'] == 'female')
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            no_history = int(request.form['family'] == "No")
            history = int(request.form['family'] == "Yes")
            caloric_food_no = int(request.form['caloric_food'] == 'No')
            caloric_food_yes = int(request.form['caloric_food'] == 'Yes')
            vegetables = int(request.form['vegetables'])
            number_meals = int(request.form['number_meals'])
            food_between_meals = int(request.form['food_between_meals'])
            smoke_no = int(request.form['smoke'] == "No")
            smoke_yes = int(request.form['smoke'] == "Yes")
            water = int(request.form['water'])
            calories_no = int(request.form['calories'] == 'No')
            calories_yes = int(request.form['calories'] == 'Yes')
            activity = int(request.form['activity'])
            technology = int(request.form['technology'])
            alcohol = int(request.form['alcohol'])
            automobile = int(request.form['transportation'] == 'automobile')
            bike = int(request.form['transportation'] == 'bike')
            motorbike = int(request.form['transportation'] == 'motorbike')
            public = int(request.form['transportation'] == 'public')
            walking = int(request.form['transportation'] == 'walking')

            data = np.array([age, height, weight, vegetables, number_meals, food_between_meals, water, activity,
                             technology, alcohol, female, male, no_history, history, caloric_food_no, caloric_food_yes,
                             smoke_no, smoke_yes, calories_no, calories_yes, automobile, bike, motorbike, public, walking]).reshape(1, -1)
            print(data)

            data_scaled = scaler.transform(data)
            print(data_scaled)

            prediction = model.predict(data_scaled) 
            print(prediction)
            predicted_label = int(np.argmax(prediction, axis=1) + 1)
            print(predicted_label)
            return render_template('result.html', name=name, predicted_label=predicted_label)
            
        except Exception as e:
            error_message = "Error occurred: " + str(e)
            return render_template('survey.html', error_message=error_message)
    else:
        return render_template('survey.html')
    
@app.route('/dashboard')
def dashboard():
    name = request.args.get('name')
    predicted_label = request.args.get('predicted_label')
    return render_template('dashboard.html', name=name, predicted_label=predicted_label)

@app.route('/result')
def result():
    name = request.args.get('name')
    predicted_label = request.args.get('predicted_label')
    return render_template('result.html', name=name, predicted_label=predicted_label)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))
    
if __name__ == '__main__':
    app.run(debug=True)