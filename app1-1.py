# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Kobieta",1:"Mężczyzna"}
ChestPainType_d = {0:"ATA",1:"NAP", 2:"ASY"}
RestingECG_d = {0:"Normal",1:"ST"}
ExerciseAngina_d = {0:"N",1:"Y"}
ST_Slope_d = {0:"Up",1:"Flat"}
FastingBS_d = {0:"0",1:"1"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Aplikacja do predykcji przeżycia pasażera Titanica")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

	with overview:
		st.title("Aplikacja do predykcji przeżycia pasażera Titanica")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x])
		ChestPainType_radio = st.radio( "ChestPainType", list(ChestPainType_d.keys()), format_func=lambda x: ChestPainType_d[x])
		RestingECG_radio = st.radio( "RestingECG", list(RestingECG_d.keys()), format_func=lambda x : RestingECG_d[x])
		ExerciseAngina_radio = st.radio( "ExerciseAngina", list(ExerciseAngina_d.keys()), format_func=lambda x : ExerciseAngina_d[x])
		ST_Slope_radio = st.radio( "ST_Slope", list(ST_Slope_d.keys()), format_func= lambda x: ST_Slope_d[x])
		FastingBS_radio = st.radio( "ST_Slope", list(FastingBS_d.keys()), format_func= lambda x: FastingBS_d[x])

	with right:
		age_slider = st.slider("Age", value=1, min_value=1, max_value=100)
		RestingBP_slider = st.slider("RestingBP", min_value=0, max_value=200)
		Cholesterol_slider = st.slider("Cholesterol", min_value=0, max_value=1000)
		MaxHR_slider = st.slider("MaxHR", min_value=0, max_value=500, step=1)
		Oldpeak_slider = st.slider("Oldpeak", min_value=0, max_value=20, step=1)

	base_data = [[age_slider, sex_radio, ChestPainType_radio, RestingBP_slider, Cholesterol_slider, FastingBS_radio, RestingECG_radio,  MaxHR_slider, ExerciseAngina_radio, Oldpeak_slider, ST_Slope_radio   ]]
	HeartDisease = model.predict(base_data)
	s_confidence = model.predict_proba(base_data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if HeartDisease[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][HeartDisease][0] * 100))

if __name__ == "__main__":
        main()
