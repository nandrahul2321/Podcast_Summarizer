from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import os
import asyncio  # 🔧 ADDED

from db import users_collection
from home import home_bp

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or 'dev-secret-key'


# -------------------- Async Safe Runner --------------------
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)
# ----------------------------------------------------------


# -------------------- Flask-Login Setup --------------------
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.session_protection = "strong"
login_manager.init_app(app)


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.password_hash = user_data['password']


@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None
    except Exception:
        return None


# -------------------- Routes --------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if len(username) < 4:
            flash('Username must be at least 4 characters', 'error')
            return redirect(url_for('signup'))

        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('signup'))

        if users_collection.find_one({'username': username}):
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))

        users_collection.insert_one({
            'username': username,
            'password': generate_password_hash(password)
        })

        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user_data = users_collection.find_one({'username': username})

        if not user_data or not check_password_hash(user_data['password'], password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))

        user = User(user_data)
        login_user(user, remember=True)

        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('home_bp.dashboard'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if len(username) < 4:
            flash('Username must be at least 4 characters', 'error')
            return redirect(url_for('forgot_password'))

        if len(new_password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('forgot_password'))

        if new_password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('forgot_password'))

        user_data = users_collection.find_one({'username': username})
        if not user_data:
            flash('Username not found', 'error')
            return redirect(url_for('forgot_password'))

        users_collection.update_one(
            {'username': username},
            {'$set': {'password': generate_password_hash(new_password)}}
        )

        flash('Password reset successfully! Please login with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot_password.html')

# -------------------- Backwards Compatibility --------------------
@app.route('/home')
@login_required
def home_redirect():
    return redirect(url_for('home_bp.dashboard'))


# -------------------- Blueprint --------------------
app.register_blueprint(home_bp)


if __name__ == '__main__':
    app.run(debug=True)
