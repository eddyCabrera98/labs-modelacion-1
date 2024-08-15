import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from simulacion import *

def uniform_fit(variable):
    fig = plt.figure()
    data=df[variable]
    a = data.min()
    b = data.max()

    # Step 2: Fit the uniform distribution
    # In a uniform distribution, the distribution is described by 'loc' (a) and 'scale' (b-a)
    loc, scale = a, b - a

    # Step 3: Plot the histogram of the data
    plt.hist(data, bins=10, density=True, alpha=0.6, color='g')

    # Plot the uniform PDF
    x = np.linspace(a, b, 100)
    plt.plot(x, uniform.pdf(x, loc, scale), 'k-', lw=2)
    plt.title(f'Fit distribucion uniforme {variable}: a = {a:.2f}, b = {b:.2f}')
    plt.xlabel(variable)
    plt.ylabel('Density')
    st.pyplot(fig) # Crea la grafica en streamlit

def triangular_fit(variable):
    fig = plt.figure()
    data = df[variable]
    left = data.min()
    right = data.max()
    mode = data.mode()[0]
    plt.hist(data, bins=10, density=True, alpha=0.6, color='g')
    x = np.linspace(left, right, 1000)
    pdf = np.where(x < mode, 2*(x-left)/((right-left)*(mode-left)), 2*(right-x)/((right-left)*(right-mode)))
    plt.plot(x, pdf, 'r-', lw=2)
    plt.title(f'Fit de Distribucion triangular {variable}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    st.pyplot(fig) # Crea la grafica en streamlit


st.title("Lab3")
st.header("Ejercicio 1")

simular1 = st.button("Iniciar simulacion", key='btn1')
if (simular1):
    mean=0.04
    std_dev=0.10
    years=17
    n_sims = 1000
    annual_contribution = 20000
    total_investment = years*annual_contribution
    # Given parameters
    final_values, accumulated_by_year_by_sim, saved_by_year_by_sim, interest_by_year_by_sim = ejercicio1(annual_contribution, mean, std_dev, years, n_sims)
    mean_final_value = np.mean(final_values)
    min_final_value = np.min(final_values)
    max_final_value = np.max(final_values)
    transposed_lists = np.array(saved_by_year_by_sim).T
    monto_rendimiento_acumulado = np.mean(np.array([np.mean(group) for group in transposed_lists]))
    st.text("Rendimiento promedio total al finalizar los 17 años")
    st.text(f"Rendimiento promedio total al finalizar los ${years} años: ${monto_rendimiento_acumulado:,.2f}")
    st.text(f"Monto promedio acumulado al finalizar los ${years} años: ${mean_final_value:,.2f}")
    st.text(f"Escenario pesimista después de ${years} años: ${min_final_value:,.2f}")
    st.text(f"Escenario optimista después de ${years} años: ${max_final_value:,.2f}")
    fig = plt.figure(figsize=(8,6))
    # Transpose the list to group values by position
    transposed_lists = np.array(saved_by_year_by_sim).T
    # Calculate the average for each position
    averages = [np.mean(group) for group in transposed_lists]
    plt.xlabel("Años")
    plt.ylabel("Gráfica de los rendimientos obtenidos por cada año")
    plt.plot(range(1, years+1), averages)
    st.pyplot(fig) # Crea la grafica en streamlit

    fig = plt.figure(figsize=(8,6))
    # Transpose the list to group values by position
    transposed_lists = np.array(interest_by_year_by_sim).T

    # Calculate the average for each position and subtract 2000 times the position
    averages = [np.mean(group) for group in transposed_lists]
    plt.xlabel("Años")
    plt.ylabel("Monto ahorrado por año")
    plt.plot(range(1, years+1), averages) 
    st.pyplot(fig) # Crea la grafica en streamlit

    fig = plt.figure(figsize=(8,6))
    transposed_lists = np.array(accumulated_by_year_by_sim).T

    # Calculate the average for each position and subtract 2000 times the position
    averages = [np.mean(group) for group in transposed_lists]
    plt.xlabel("Años")
    plt.ylabel("Grafica del monto acumulado por cada año dentro de los 17 años.")
    plt.plot(range(1, years+1), averages) 
    st.pyplot(fig) # Crea la grafica en streamlit


st.header("Ejercicio 2")
st.subheader("Regresion Lineal")
df = pd.read_csv('advertising.csv',  index_col=0)
variables = ['TV', 'Radio', 'Newspaper']
X = df[variables]
Y = df['Sales']
model = LinearRegression()
model.fit(X, Y)
st.text(f"Coefficients: {model.coef_}")
st.text(f"Intercept: {model.intercept_}")
st.subheader("Distribuciones")
for var in variables:
    fig = plt.figure()
    plt.hist(df[var], bins=10, density=True)
    df[var].plot(kind='kde')
    plt.xlabel(var)
    plt.ylabel("Densidad")
    plt.title(f"Distribucion de probabilidad de {var}")
    st.pyplot(fig) # Crea la grafica en streamlit
summary = df.describe()
st.text(summary)


uniform_fit("TV")
uniform_fit("Radio")
triangular_fit("Newspaper")


n_sims_2 = st.slider("Cantidad de simulaciones", min_value=500, max_value=5000, value=1000)
simular2 = st.button("Iniciar simulacion", key='btn2')
if (simular2):
    df_results = ejercicio2(n_sims_2, 200, df, model)
    tv=round(df_results['TV'].mean(), 2)
    radio=round(df_results['Radio'].mean(), 2)
    newspaper=round(df_results['Newspaper'].mean(), 2)
    st.subheader("Valores de inversión de cada tipo de publicidad maximizando el valor de ventas")
    st.text(f"TV: {tv}")
    st.text(f"Radio: {radio}")
    st.text(f"Newspaper: {newspaper}")
    st.subheader("Porcentaje")
    st.text(f"TV: {tv/(tv+radio+newspaper)*100:.2f}%")
    st.text(f"Radio: {radio/(tv+radio+newspaper)*100:.2f}%")
    st.text(f"Newspaper:  {newspaper/(tv+radio+newspaper)*100:.2f}%")
