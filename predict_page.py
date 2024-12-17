import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_Brand = data["Car Brand"]
le_Model = data["Car Model"]

def show_predict_page():
    st.title("CarVal")

    st.write("""### Hey there! Tell us a little about your car so we can predict its price""")

Brands = ('Nissan',
           'Toyota', 
           'Ford', 
           'Chevrolet', 
           'Honda',
             'Kia', 
             'Hyundai')


Models = (
"Corolla",  
"Civic",  
"Mustang",  
"Silverado",  
"Altima",  
"3 Series",  
"C-Class",  
"Golf",  
"A4",  
"Elantra",  
"Optima",  
"CX-5",  
"Outback",  
"RX",  
"XF",  
"Range Rover",  
"911",  
"488",  
"Aventador",  
"XC90",  
"Outlander",  
"208",  
"Clio",  
"500",  
"Wrangler",  
"Challenger",  
"300",  
"Enclave",  
"Escalade",  
"Sierra",  
"Model S",  
"Q50",  
"TLX",  
"Navigator",  
"Ghibli",  
"Continental GT",  
"Phantom",  
"Giulia",  
"DB11",  
"720S")

Specs = ("GCC",
         "Imported")
Year = (1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)

Brand = st.selectbox("Brand", Brands)

Model = st.selectbox("Model", Models)
Age = st.selectbox("Year of production", Year)
Specifications = st.selectbox("Specifications", Specs)
Mileage = st.slider("Mileage", 0, 1000000, 0)

ok = st.button("Calculate Price")

if ok:
    X = np.array([[Brand, Model, Age, Mileage, 0, 0, 0, 0]])
    X[:, 0] = le_Brand.transform(X[:,0])
    X[:, 1] = le_Model.transform(X[:,1])
    X = X.astype(float)


    Price = regressor_loaded.predict(X)
    st.subheader(f"The estmiated price is AED{Price[0]:.2f}")