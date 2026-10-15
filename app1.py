import datetime

import joblib
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 820px;}
        div[data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.08);
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 12px;
            padding: 14px 18px;
        }
        div.stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 10px;
        }
        .price-card {
            text-align: center;
            padding: 1.6rem 1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #1f7a4d 0%, #2fa36b 100%);
            color: white;
            margin-top: 1rem;
        }
        .price-card .label {font-size: 0.95rem; opacity: 0.9;}
        .price-card .value {font-size: 2.4rem; font-weight: 700; margin-top: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("XGboost_Regressor_Model.pkl")


model = load_model()

# ---------- Data ----------
car_names = [
    'Audi A3 Cabriolet', 'Audi A4 1.8', 'Audi A4 2.0', 'Audi A6 2.0',
    'Audi A8', 'Audi Q3 2.0', 'Audi Q5 2.0', 'Audi Q7', 'BMW 3 Series',
    'BMW 5 Series', 'BMW 7 Series', 'BMW X1', 'BMW X1 sDrive20d',
    'BMW X1 xDrive20d', 'Chevrolet Beat', 'Chevrolet Beat Diesel',
    'Chevrolet Beat LS', 'Chevrolet Beat LT', 'Chevrolet Beat PS',
    'Chevrolet Cruze LTZ', 'Chevrolet Enjoy', 'Chevrolet Enjoy 1.4',
    'Chevrolet Sail 1.2', 'Chevrolet Sail UVA', 'Chevrolet Spark',
    'Chevrolet Spark 1.0', 'Chevrolet Spark LS', 'Chevrolet Spark LT',
    'Chevrolet Tavera LS', 'Chevrolet Tavera Neo', 'Datsun GO T',
    'Datsun Go Plus', 'Datsun Redi GO', 'Fiat Linea Emotion',
    'Fiat Petra ELX', 'Fiat Punto Emotion', 'Force Motors Force',
    'Force Motors One', 'Ford EcoSport', 'Ford EcoSport Ambiente',
    'Ford EcoSport Titanium', 'Ford EcoSport Trend',
    'Ford Endeavor 4x4', 'Ford Fiesta', 'Ford Fiesta SXi', 'Ford Figo',
    'Ford Figo Diesel', 'Ford Figo Duratorq', 'Ford Figo Petrol',
    'Ford Fusion 1.4', 'Ford Ikon 1.3', 'Ford Ikon 1.6',
    'Hindustan Motors Ambassador', 'Honda Accord', 'Honda Amaze',
    'Honda Amaze 1.2', 'Honda Amaze 1.5', 'Honda Brio', 'Honda Brio V',
    'Honda Brio VX', 'Honda City', 'Honda City 1.5', 'Honda City SV',
    'Honda City VX', 'Honda City ZX', 'Honda Jazz S', 'Honda Jazz VX',
    'Honda Mobilio', 'Honda Mobilio S', 'Honda WR V', 'Hyundai Accent',
    'Hyundai Accent Executive', 'Hyundai Accent GLE',
    'Hyundai Accent GLX', 'Hyundai Creta', 'Hyundai Creta 1.6',
    'Hyundai Elantra 1.8', 'Hyundai Elantra SX', 'Hyundai Elite i20',
    'Hyundai Eon', 'Hyundai Eon D', 'Hyundai Eon Era',
    'Hyundai Eon Magna', 'Hyundai Eon Sportz', 'Hyundai Fluidic Verna',
    'Hyundai Getz', 'Hyundai Getz GLE', 'Hyundai Getz Prime',
    'Hyundai Grand i10', 'Hyundai Santro', 'Hyundai Santro AE',
    'Hyundai Santro Xing', 'Hyundai Sonata Transform', 'Hyundai Verna',
    'Hyundai Verna 1.4', 'Hyundai Verna 1.6', 'Hyundai Verna Fluidic',
    'Hyundai Verna Transform', 'Hyundai Verna VGT',
    'Hyundai Xcent Base', 'Hyundai Xcent SX', 'Hyundai i10',
    'Hyundai i10 Era', 'Hyundai i10 Magna', 'Hyundai i10 Sportz',
    'Hyundai i20', 'Hyundai i20 Active', 'Hyundai i20 Asta',
    'Hyundai i20 Magna', 'Hyundai i20 Select', 'Hyundai i20 Sportz',
    'Jaguar XE XE', 'Jaguar XF 2.2', 'Jeep Wrangler Unlimited',
    'Land Rover Freelander', 'Mahindra Bolero DI',
    'Mahindra Bolero Power', 'Mahindra Bolero SLE',
    'Mahindra Jeep CL550', 'Mahindra Jeep MM', 'Mahindra KUV100',
    'Mahindra KUV100 K8', 'Mahindra Logan', 'Mahindra Logan Diesel',
    'Mahindra Quanto C4', 'Mahindra Quanto C8', 'Mahindra Scorpio',
    'Mahindra Scorpio 2.6', 'Mahindra Scorpio LX',
    'Mahindra Scorpio S10', 'Mahindra Scorpio S4',
    'Mahindra Scorpio SLE', 'Mahindra Scorpio SLX',
    'Mahindra Scorpio VLX', 'Mahindra Scorpio Vlx',
    'Mahindra Scorpio W', 'Mahindra TUV300 T4', 'Mahindra TUV300 T8',
    'Mahindra Thar CRDe', 'Mahindra XUV500', 'Mahindra XUV500 W10',
    'Mahindra XUV500 W6', 'Mahindra XUV500 W8', 'Mahindra Xylo D2',
    'Mahindra Xylo E4', 'Mahindra Xylo E8', 'Maruti Suzuki 800',
    'Maruti Suzuki A', 'Maruti Suzuki Alto', 'Maruti Suzuki Baleno',
    'Maruti Suzuki Celerio', 'Maruti Suzuki Ciaz',
    'Maruti Suzuki Dzire', 'Maruti Suzuki Eeco',
    'Maruti Suzuki Ertiga', 'Maruti Suzuki Esteem',
    'Maruti Suzuki Estilo', 'Maruti Suzuki Maruti',
    'Maruti Suzuki Omni', 'Maruti Suzuki Ritz', 'Maruti Suzuki S',
    'Maruti Suzuki SX4', 'Maruti Suzuki Stingray',
    'Maruti Suzuki Swift', 'Maruti Suzuki Versa',
    'Maruti Suzuki Vitara', 'Maruti Suzuki Wagon', 'Maruti Suzuki Zen',
    'Mercedes Benz A', 'Mercedes Benz B', 'Mercedes Benz C',
    'Mercedes Benz GLA', 'Mini Cooper S', 'Mitsubishi Lancer 1.8',
    'Mitsubishi Pajero Sport', 'Nissan Micra XL', 'Nissan Micra XV',
    'Nissan Sunny', 'Nissan Sunny XL', 'Nissan Terrano XL',
    'Nissan X Trail', 'Renault Duster', 'Renault Duster 110',
    'Renault Duster 110PS', 'Renault Duster 85', 'Renault Duster 85PS',
    'Renault Duster RxL', 'Renault Kwid', 'Renault Kwid 1.0',
    'Renault Kwid RXT', 'Renault Lodgy 85', 'Renault Scala RxL',
    'Skoda Fabia', 'Skoda Fabia 1.2L', 'Skoda Fabia Classic',
    'Skoda Laura', 'Skoda Octavia Classic', 'Skoda Rapid Elegance',
    'Skoda Superb 1.8', 'Skoda Yeti Ambition', 'Tata Aria Pleasure',
    'Tata Bolt XM', 'Tata Indica', 'Tata Indica V2', 'Tata Indica eV2',
    'Tata Indigo CS', 'Tata Indigo LS', 'Tata Indigo LX',
    'Tata Indigo Marina', 'Tata Indigo eCS', 'Tata Manza',
    'Tata Manza Aqua', 'Tata Manza Aura', 'Tata Manza ELAN',
    'Tata Nano', 'Tata Nano Cx', 'Tata Nano GenX', 'Tata Nano LX',
    'Tata Nano Lx', 'Tata Sumo Gold', 'Tata Sumo Grande',
    'Tata Sumo Victa', 'Tata Tiago Revotorq', 'Tata Tiago Revotron',
    'Tata Tigor Revotron', 'Tata Venture EX', 'Tata Vista Quadrajet',
    'Tata Zest Quadrajet', 'Tata Zest XE', 'Tata Zest XM',
    'Toyota Corolla', 'Toyota Corolla Altis', 'Toyota Corolla H2',
    'Toyota Etios', 'Toyota Etios G', 'Toyota Etios GD',
    'Toyota Etios Liva', 'Toyota Fortuner', 'Toyota Fortuner 3.0',
    'Toyota Innova 2.0', 'Toyota Innova 2.5', 'Toyota Qualis',
    'Volkswagen Jetta Comfortline', 'Volkswagen Jetta Highline',
    'Volkswagen Passat Diesel', 'Volkswagen Polo',
    'Volkswagen Polo Comfortline', 'Volkswagen Polo Highline',
    'Volkswagen Polo Highline1.2L', 'Volkswagen Polo Trendline',
    'Volkswagen Vento Comfortline', 'Volkswagen Vento Highline',
    'Volkswagen Vento Konekt', 'Volvo S80 Summum',
]

# The company is always the first word of the car name, so it can be
# derived instead of being maintained as a separate list.
company_names = sorted({n.split()[0] for n in car_names})

CURRENT_YEAR = datetime.date.today().year


def format_inr(amount: float) -> str:
    """Format a number with Indian digit grouping (e.g. 1,23,456)."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    s = str(abs(amount))
    if len(s) <= 3:
        return f"{sign}₹{s}"
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join(groups)},{tail}"


# ---------- Header ----------
st.title("🚗 Car Price Predictor")
st.caption("Estimate the resale value of a used car from its make, model, age and usage.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This tool uses an XGBoost regression model trained on used-car "
        "listings. Treat the result as an estimate, not a valuation."
    )
    st.markdown("**How to use**")
    st.markdown(
        "1. Pick the company, then the model\n"
        "2. Enter the year and kilometres driven\n"
        "3. Choose the fuel type and click Predict"
    )

# ---------- Inputs ----------
st.subheader("Car details")

col1, col2 = st.columns(2)
with col1:
    company_name = st.selectbox("Company", company_names, index=company_names.index("Maruti"))
with col2:
    models_for_company = [n for n in car_names if n.split()[0] == company_name]
    name = st.selectbox("Car model", models_for_company)

col3, col4, col5 = st.columns(3)
with col3:
    year = st.number_input(
        "Year of purchase",
        min_value=1995,
        max_value=CURRENT_YEAR,
        value=2015,
        step=1,
    )
with col4:
    km_driven = st.number_input(
        "Kilometres driven",
        min_value=0,
        max_value=1_000_000,
        value=40_000,
        step=1_000,
    )
with col5:
    fuel_type = st.radio("Fuel type", ["Petrol", "Diesel", "LPG"])

st.write("")
predict_clicked = st.button("Predict price", type="primary")

# ---------- Prediction ----------
if predict_clicked:
    # Same input structure as the original app, including reset_index(),
    # which adds the "index" column the model pipeline was built with.
    input_df = pd.DataFrame(
        {
            "name": [name],
            "company": [company_name],
            "year": [year],
            "kms_driven": [km_driven],
            "fuel_type": [fuel_type],
        }
    ).reset_index()

    try:
        with st.spinner("Calculating..."):
            prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.markdown(
            f"""
            <div class="price-card">
                <div class="label">Estimated resale value</div>
                <div class="value">{format_inr(prediction)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Car age", f"{CURRENT_YEAR - int(year)} yrs")
        m2.metric("Kms driven", f"{int(km_driven):,}")
        m3.metric("Fuel", fuel_type)

        with st.expander("Input summary"):
            st.dataframe(
                input_df.drop(columns="index").rename(
                    columns={
                        "name": "Car",
                        "company": "Company",
                        "year": "Year",
                        "kms_driven": "Kms driven",
                        "fuel_type": "Fuel",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )