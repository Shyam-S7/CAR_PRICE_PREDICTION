from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/form", methods=["GET"])
def show_form():
    return render_template(
        "form.html",
        brands=["Audi", "BMW", "Mercedes", "Volkswagen"],
        models_dict={
            "Audi": ["A3", "A4", "Q7"],
            "BMW": ["320i", "X5", "M3"],
            "Mercedes": ["C200", "E220", "GLA"],
            "Volkswagen": ["Polo", "Golf", "Passat"]
        },
        colors=["Black", "White", "Silver", "Blue", "Red"],
        years=list(range(2000, 2025)),
        transmissions=["Manual", "Automatic"],
        fuel_types=["Petrol", "Diesel", "Hybrid", "Electric"]
    )


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    # Collect inputs from the form
    data = CustomData(
        brand=request.form.get("brand"),
        model=request.form.get("model_name"),
        color=request.form.get("color"),
        year=int(request.form.get("year")),
        power_kw=float(request.form.get("power_kw")),
        power_ps=float(request.form.get("power_ps")),
        transmission_type=request.form.get("transmission"),
        fuel_type=request.form.get("fuel_type"),
        fuel_consumption_l_100km=float(request.form.get("fuel_100km")),
        fuel_consumption_g_km=float(request.form.get("fuel_g_km")),
        mileage_in_km=float(request.form.get("mileage")),
    )

    # Convert to dataframe
    final_new_data = data.get_data_as_dataframe()

    # Make prediction
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_new_data)

    results = round(pred[0], 2)

    return render_template("results.html", final_result=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
