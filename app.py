from flask import Flask, render_template, redirect, request
import predict_titanic
# Create an instance of Flask
app = Flask(__name__)


# Route to render index.html template using data from Mongo
@app.route("/")
def home():
    ticket_price = ""
    gender_mf = ""
    # Return template and data
    return render_template("index.html",ticketPrice=ticket_price, genderMF=gender_mf)

@app.route("/submit", methods=["POST"])
def form():
    ticket_price = request.form["price"]
    gender_mf = int(request.form["gender"])
    predict = predict_titanic.predict(gender_mf)
    # Right here we"ll do gender next
    # Return template and data
    return render_template("index.html",ticketPrice=ticket_price, genderMF=predict)




if __name__ == "__main__":
    app.run(debug=True)

# This is Lesson 3 Unit 12 Activity 10