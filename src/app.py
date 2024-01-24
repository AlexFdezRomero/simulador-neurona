import streamlit as st
import pandas as pd 
import numpy as np
import math
from neuron import Neuron

st.set_page_config(layout="wide")

st.image('./data/neuron.webp', width=350)

st.title("Simulador de neurona")

st.subheader("ALEJANDRO FR")

num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", min_value=1, max_value=10, value=1, step=1, key="num")

st.subheader("Pesos")

columnas_w = st.columns(num)

w = []

for i in range(num):

    w.append(columnas_w[i].number_input(f"$w_{i}$", key=f"w{i}"))

st.text(f"w = {w}")

st.subheader("Entradas")

columnas_x = st.columns(num)

x = []

for i in range(num):

    x.append(columnas_x[i].number_input(f"$x_{i}$", key=f"x{i}"))

st.text(f"x = {x}")

col = st.columns(2)

col[0].subheader("Sesgo")

b = col[0].number_input('Introduzca el valor del sesgo', key="b")

col[1].subheader("Función de acitvación")

f = str(col[1].selectbox('Introduzca el valor del sesgo', ("ReLU", "TanH", "Sigmoide"), key="f"))
        
if st.button("Calcular la salida"):
   n1 = Neuron(weights=w, bias=b, func=f)
   y = n1.run(input_data=x)
   st.text(f"La salida de la neurona es: {y}")

    
