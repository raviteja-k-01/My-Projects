from flask import Flask, render_template, request, redirect, session
from database.tables import create_tables
from database.register import Register
from database.login import Login
import balance as b
import deposite as dp
import withdrawal as w
import transfer as t
import data as d
from exception import *

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Create DB tables at start
create_tables()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            account_no = request.form["account_no"]
            password = request.form["password"]
            username = request.form["username"]
            email = request.form["email"]
            amount = request.form["amount"]

            reg = Register()
            result = reg.register(
                account_no=account_no,
                password=password,
                username=username,
                email=email,
                deposite_amount=amount
            )
            return render_template("register.html", message=result)

        except Exception as e:
            return render_template("register.html", message=f"Error: {e}")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        account_no = request.form["account_no"]
        password = request.form["password"]

        try:
            log_obj = Login()
            login_status = log_obj.login(account_no, password)
            if login_status:
                session["user"] = account_no
                return redirect("/dashboard")
            else:
                return render_template("login.html", message="Invalid login")

        except Exception as e:
            return render_template("login.html", message=f"Error: {e}")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html")


@app.route("/balance")
def balance():
    if "user" not in session:
        return redirect("/login")
    bal = b.balance(account=session["user"])
    return render_template("balance.html", balance=bal)


@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        amount = int(request.form["amount"])
        try:
            result = w.withdraw(account=session["user"], amount=amount)
            return render_template("withdraw.html", message=result)
        except Exception as e:
            return render_template("withdraw.html", message=f"Error: {e}")

    return render_template("withdraw.html")


@app.route("/deposit", methods=["GET", "POST"])
def deposit():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        amount = int(request.form["amount"])
        try:
            result = dp.deposite(account=session["user"], amount=amount)
            return render_template("deposit.html", message=result)
        except Exception as e:
            return render_template("deposit.html", message=f"Error: {e}")

    return render_template("deposit.html")


@app.route("/transfer", methods=["GET", "POST"])
def transfer():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        to_acc = int(request.form["to_account"])
        amount = int(request.form["amount"])

        try:
            result = t.transfer(session["user"], to_acc, amount)
            return render_template("transfer.html", message=result)
        except Exception as e:
            return render_template("transfer.html", message=f"Error: {e}")

    return render_template("transfer.html")


    
@app.route("/ministatement")
def ministatement():
    if "user" not in session:
        return redirect("/login")
    stmt = d.ministatement(account=session["user"])
    return render_template("ministatement.html", statement=stmt)


# @app.route("/ministatement")
# def ministatement():
#     if "user" not in session:
#         return redirect("/login")
#     stmt = d.ministatement(account=session["user"])
#     return render_template("ministatement.html", statement=stmt)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True)
