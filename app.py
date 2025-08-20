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
        
        brands=['alfa-romeo', 'aston-martin', 'audi', 'bentley', 'bmw', 'cadillac',
 'chevrolet', 'chrysler', 'citroen', 'dacia', 'daewoo', 'daihatsu',
 'dodge', 'ferrari', 'fiat', 'ford', 'honda', 'hyundai', 'infiniti',
 'isuzu', 'jaguar', 'jeep', 'kia', 'lamborghini', 'lancia',
 'land-rover', 'lada', 'maserati', 'mazda'],

        
        
        
        
        
models_dict = {
    "alfa-romeo": ['Alfa Romeo', 'Alfa Romeo 147', 'Alfa Romeo 145', 'Alfa Romeo 146', 'Alfa Romeo 155',
                    'Alfa Romeo 156', 'Alfa Romeo 159', 'Alfa Romeo 164', 'Alfa Romeo 166', 'Alfa Romeo 4C',
                    'Alfa Romeo 8C', 'Alfa Romeo Alfa 6', 'Alfa Romeo Brera', 'Alfa Romeo Giulia',
                    'Alfa Romeo Giulietta', 'Alfa Romeo GT', 'Alfa Romeo GTV', 'Alfa Romeo MiTo',
                    'Alfa Romeo Spider', 'Alfa Romeo Sportwagon', 'Alfa Romeo Stelvio', 'Alfa Romeo Tonale'],
    
    "aston-martin": ['Aston Martin', 'Aston Martin DB7', 'Aston Martin DB9', 'Aston Martin DB11',
                      'Aston Martin DBX', 'Aston Martin DBS', 'Aston Martin Lagonda', 'Aston Martin Rapide',
                      'Aston Martin Vantage', 'Aston Martin V8', 'Aston Martin Vanquish', 'Aston Martin Virage'],

    "audi": ['50', '80', 'A1', 'A2', 'A3', 'A4', 'A4 allroad', 'A5', 'A6', 'A6 allroad', 'A7', 'A8',
             'Allroad', 'Cabriolet', 'e-tron', 'e-tron GT', 'Q2', 'Q3', 'Q4 e-tron', 'Q5', 'Q7', 'Q8',
             'Q8 e-tron', 'RS', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'RS Q3', 'RS Q8', 'S1', 'S3',
             'S4', 'S5', 'S6', 'S7', 'S8', 'SQ2', 'SQ5', 'SQ7', 'SQ8', 'TTS', 'TT', 'TT RS', 'QUATTRO'],

    "bmw": ['114', '116', '118', '120', '123', '125', '128', '130', '135', '140', '216', '218',
            '220', '223', '225', '228', '230', '235', '240', '318', '320', '323', '325', '328',
            '330', '335', '340', '418', '420', '425', '428', '430', '435', '440', '518', '520',
            '523', '525', '528', '530', '535', '540', '550', '620', '630', '635', '640', '645',
            '728', '735', '740', '745', '750', '760', '840', '850', 'i3', 'i4', 'i5', 'i7', 'i8',
            'iX', 'iX1', 'iX3', 'iX55', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M8', 'M550', 'M850',
            'X1', 'X2', 'X2 M', 'X3', 'X3 M', 'X4', 'X4 M', 'X5', 'X5 M', 'X6', 'X6 M', 'X7',
            'X7 M', 'XM', 'Z3', 'Z3 M', 'Z4', 'Z4 M', '1er M Coupé'],

    "cadillac": ['ATS', 'BLS', 'CT6', 'CTS', 'Eldorado', 'Escalade', 'Seville', 'SRX', 'STS',
                 'XT4', 'XT5', 'XT6'],

    "chevrolet": ['2500', 'Aveo', 'Blazer', 'Bolt', 'Camaro', 'Captiva', 'Chevy Van', 'Colorado',
                  'Corvette', 'C1500', 'Cruze', 'Express', 'Kalos', 'Matiz', 'Orlando', 'Silverado',
                  'Suburban', 'Tahoe', 'Trailblazer', 'Trax'],

    "chrysler": ['200', 'Pacifica', 'Ram Van'],

    "dacia": ['Dokker', 'Duster', 'Jogger', 'Lodgy', 'Logan', 'Pick Up', 'Sandero', 'Spring'],

    "daihatsu": ['Applause', 'Charade', 'Copen', 'Cuore', 'Materia', 'Move', 'Sirion', 'Terios',
                 'Trevis', 'YRV'],

    "dodge": ['Avenger', 'Caliber', 'Challenger', 'Charger', 'Durango', 'Grand Caravan', 'Journey', 'Nitro', 'RAM'],

    "daewoo": ['Evanda', 'Espero', 'Kalos', 'Lanos', 'Matiz', 'Nubira', 'Rezzo', 'Tacuma'],

    "fiat": ['124 Spider', '500', '500C', '500L', '500X', '500e', '595 Abarth', 'Bravo', 'Croma',
             'Doblo', 'Ducato', 'E-Doblo', 'Fiorino', 'Freemont', 'Fullback', 'Grande Punto', 'Idea',
             'Linea', 'Multipla', 'New Panda', 'Panda', 'Punto', 'Punto Evo', 'Qubo', 'Scudo',
             'Seicento', 'Stilo', 'Strada', 'Talento', 'Ulysse'],

    "ford": ['B-Max', 'Bronco', 'C-Max', 'Courier', 'Crown', 'EcoSport', 'Edge', 'Escort',
             'E-Transit', 'Expedition', 'F 150', 'F 250', 'F 350', 'Fiesta', 'Focus', 'Focus C-Max',
             'Focus CC', 'Fusion', 'Galaxy', 'GT', 'Grand C-Max', 'Grand Tourneo', 'Ka/Ka+', 'Kuga',
             'M', 'Maverick', 'Mustang', 'Mustang Mach-E', 'Mondeo', 'Probe', 'Puma', 'Ranger',
             'Ranger Raptor', 'S-Max', 'Streetka', 'Taurus', 'Tourneo', 'Tourneo Connect',
             'Tourneo Courier', 'Tourneo Custom', 'Transit', 'Transit Bus', 'Transit Connect',
             'Transit Courier', 'Transit Custom', 'Windstar', 'Explorer'],

    "hyundai": ['ACCENT', 'Atos', 'BAYON', 'Coupe', 'ELANTRA', 'Genesis', 'Genesis Coupe', 'Getz',
                'Grand Santa Fe', 'H 350', 'H-1', 'IONIQ', 'IONIQ 5', 'IONIQ 6', 'i10', 'i20', 'i30',
                'i40', 'iX20', 'iX35', 'iX55', 'KONA', 'KONA Elektro', 'Matrix', 'NEXO', 'SANTA FE',
                'STARIA', 'SONATA', 'Terracan', 'TUCSON', 'VELOSTER'],

    "honda": ['Accord', 'CR-V', 'Civic', 'HR-V', 'Insight', 'Jazz', 'NSX', 'Odyssey', 'Stream', 'e'],

    "infiniti": ['EX30', 'EX35', 'EX37', 'FX', 'G37', 'M30', 'M35', 'Q30', 'Q50', 'Q60',
                 'Q70', 'QX30', 'QX50', 'QX60', 'QX70', 'QX80'],

    "isuzu": ['D-Max', 'Trooper'],

    "jeep": ['Avenger', 'Cherokee', 'Compass', 'Commander', 'Gladiator', 'Grand Cherokee',
             'Patriot', 'Renegade', 'Wagoneer', 'Wrangler'],

    "jaguar": ['E-Pace', 'F-Pace', 'F-Type', 'I-Pace', 'X-Type', 'XJ', 'XK', 'XKR', 'XF', 'XE'],

    "lada": ['111', '4x4', 'Granta', 'Kalina', 'Niva', 'Nova', 'Priora', 'Taiga', 'Urban', 'Vesta'],

    "lamborghini": ['Aventador', 'Diablo', 'Gallardo', 'Huracan', 'Murciélago'],

    "land-rover": ['Land Rover', 'Land Rover Defender', 'Land Rover Discovery', 'Land Rover Discovery Sport',
                    'Land Rover Freelander', 'Land Rover Range Rover', 'Land Rover Range Rover Evoque',
                    'Land Rover Range Rover Sport', 'Land Rover Range Rover Velar'],

    "mazda": ['2', '3', '5', '6', '323', 'BT-50', 'CX-3', 'CX-5', 'CX-7', 'CX-9', 'CX-30', 'MX-5'],

    "maserati": ['3200', '4200', 'Coupe', 'Ghibli', 'GranCabrio', 'GranSport', 'GranTurismo',
                 'MC20', 'Quattroporte', 'Spyder'],

    "citroen": ['Ami', 'C1', 'C2', 'C3', 'C3 Aircross', 'C3 Picasso', 'C4', 'C4 Aircross', 'C4 Cactus',
                'C4 Picasso', 'C4 SpaceTourer', 'C5', 'C5 Aircross', 'C5 X', 'C6', 'C8', 'C-Crosser',
                'DS', 'DS3', 'DS4', 'DS5', 'E-C4 Electric', 'E-C4 X', 'Grand C4 Picasso', 'Grand C4 SpaceTourer',
                'Jumper', 'Jumpy', 'Nemo', 'Spacetourer', 'Xsara', 'Xsara Picasso'],

    "kia": ['Ceed SW / cee\'d SW', 'EV6', 'Niro', 'Optima', 'Picanto', 'ProCeed / pro_cee\'d',
            'Rio', 'Shuma', 'Soul', 'Sportage', 'Stinger', 'Stonic', 'Venga', 'XCeed', 'e-Niro'],

    "lancia": ['Delta', 'Dedra', 'Flavia', 'Kappa', 'Lybra', 'MUSA', 'Phedra', 'Thema', 'Voyager', 'Y', 'Ypsilon', 'ZETA'],

    "ferrari": ['296', '348', '360', '430 Scuderia', 'F12', 'F355', 'F430', 'F8 Spider', 'F8 Tributo',
                '456', '488', '512', '550', '575', '599', '612', 'California', 'FF', 'GTC4 Lusso',
                'Portofino', 'Roma', 'SF90 Spider', 'SF90 Stradale'],

    "bentley": ['Arnage', 'Azure', 'Continental', 'Continental GT', 'Continental GTC',
                'Flying Spur', 'Mulsanne', 'Turbo R']
},

       
        
        colors=['black', 'grey', 'red', 'white', 'blue', 'silver', 'brown',
       'green', 'orange', 'gold', 'yellow', 'violet', 'bronze', 'beige',
       ],
        
        years=list(range(2000, 2025)),
        transmissions=['Manual', 'Automatic', 'Unknown', 'Semi-automatic'],
        
        fuel_types=['Petrol', 'Diesel', 'Hybrid', 'Electric', 'LPG', 'Diesel Hybrid',
       'Other', 'CNG', 'Unknown', 'Ethanol', 'Hydrogen']
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
