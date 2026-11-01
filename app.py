import streamlit as st

# Título de la app
st.title("¡Hola Mundo desde Ubuntu! 🐧🚀")

# Mensaje de bienvenida
st.write("Esta es mi primera aplicación de Streamlit desarrollada localmente en Ubuntu.")

# Componente interactivo
nombre = st.text_input("Ingresa tu nombre:")

if nombre:
    st.success(f"¡Hola, {nombre}! Tu entorno local está funcionando perfectamente.")