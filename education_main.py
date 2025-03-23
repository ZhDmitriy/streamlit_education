import streamlit as st 
import pandas as pd 
import os
from io import BytesIO
import numpy as np
from datetime import datetime
import psycopg2
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

def read_insert_user_file(user_data: pd.DataFrame):

    user_data.columns = ['subject', 'number_of_tasks', 'number_task', 'difficulty_level', 
                         'min_time_on_solving', 'max_time_on_solving', 'answer', 'right_answer', 
                         'points_scored', 'max_points_scored', 'satisfaction']

    connection = psycopg2.connect(
            host='localhost',
            user='newuser',
            password='postgres',
            database='FactTable'
        )
    cursor = connection.cursor()
    
    for valueIndex in user_data.itertuples():

        print(valueIndex)

        cursor.execute("""
            INSERT INTO EducationDatamart (subject, number_of_tasks, number_task, difficulty_level, min_time_on_solving, 
                       max_time_on_solving, аnswer, right_answer, points_scored, max_points_scored, 
                       satisfaction, load_dttm)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (valueIndex.subject, valueIndex.number_of_tasks, valueIndex.number_task, valueIndex.difficulty_level, 
              valueIndex.min_time_on_solving, valueIndex.max_time_on_solving, valueIndex.answer, valueIndex.right_answer,
              valueIndex.points_scored, valueIndex.max_points_scored, valueIndex.satisfaction, datetime.now()
              ))
    
    connection.commit()


def read_database_data()-> pd.DataFrame: 
    connection = psycopg2.connect(
        host='localhost',
        user='newuser',
        password='postgres',
        database='FactTable'
    )
    education_datamart = pd.read_sql(sql="""
        SELECT * 
        FROM EducationDatamart
        WHERE load_dttm::date >= now()::date - interval '10 days'
    """, con=connection)
    connection.close()
    return education_datamart


st.set_page_config(page_title="Анализ ошибок при выполнении тестов ЕГЭ", layout="wide")
st.title("Приложение-тренажер для помощи с подготовкой к ЕГЭ по информатике")
st.subheader("Цели приложения: ")
st.write("1. Помочь проанализировать свои ошибки при выполнении тестов.")
st.write("2. Построить предиктивную аналитику для анализа будущих результатов с учетом текущих ошибок в тестах.")

base_page, current_data_db, analytic_page, predictive_analytic_page = st.tabs(["Главная страница", "Текущие данные в БД", "Аналитика", "Прогноз решения задач"])

with base_page:
    st.subheader("Загрузите файл согласно примерам: ")
    st.image(os.path.join(os.getcwd(), "static", "ExampleTest.png"), width=850)
    st.write(" ")
    st.image(os.path.join(os.getcwd(), "static", "ReportFields.png"), width=350)
    col1, col2 = st.columns([2, 3])
    with col1:
        uploaded_file = st.file_uploader("Выберите Excel-файл: ", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.balloons()
        user_data = pd.read_excel(uploaded_file)
        st.write(" ")
        st.write("Содержимое файла:")
        st.write(" ")
        st.dataframe(user_data)
        read_insert_user_file(user_data)
    else: 
        st.write("Пожалуйста, загрузите Excel-файл.")

with current_data_db: 
    st.subheader("Посмотреть текущие данные в базе данных за 10 дней")
    pressed = st.button("Посмотреть")
    if pressed:
        st.dataframe(read_database_data())

with analytic_page:
    
    user_data = read_database_data()
    st.subheader("Время, затраченное на решение задач")
    st.line_chart(user_data["min_time_on_solving"], width=350)

    st.subheader("Оценка удовлетворенности по уровням сложности")
    st.bar_chart(user_data.groupby("difficulty_level")["max_points_scored"].mean(), width=350)

    st.subheader("Распределение баллов по уровням сложности")
    st.area_chart(user_data.groupby("difficulty_level")["satisfaction"].mean(), width=350)

    st.subheader("Количество заданий по уровням сложности")
    level_counts = user_data["difficulty_level"].value_counts()
    st.bar_chart(level_counts)

def new_func(train, X, y, model):
    model.fit(train[X], train[y])

with predictive_analytic_page: 

    user_data = read_database_data()
    user_data['right_answer_status'] = (user_data['аnswer'] == user_data['right_answer']).astype(int)
    
    X = ['subject', 'number_of_tasks', 'number_task', 'difficulty_level', 
        'min_time_on_solving', 'max_time_on_solving', 'аnswer',
        'points_scored', 'max_points_scored', 'satisfaction', 'right_answer_status']
    y = ['right_answer_status']
    cat_features = ['subject', 'difficulty_level', 'number_task', 'number_of_tasks', 'аnswer', 'right_answer_status']
    
    pressed = st.button("Обучить модель")
    if pressed:
        model = CatBoostRegressor(cat_features=cat_features, eval_metric='MAPE')
        st.write("Модель обучается на данных в БД ... ")
        model.fit(X=user_data[X], y=user_data[y])
        st.write("Процесс обучения закончен ")
        user_data['predict_value'] = list(model.predict(user_data[X])*100)
        st.dataframe(user_data)
        st.subheader("Вероятность решения задач на экзамене")
        st.bar_chart(user_data.groupby("number_task")["predict_value"].mean(), width=350)


# st.divider()

# st.image(os.path.join(os.getcwd(), "static", "Apache Airflow.png"), width=50)

# df = pd.DataFrame({
#     "Name": ['Alice', 'Bob'],
#     "Age": [25, 32], 
#     "Occupation": ['Engineer', 'Doctor']
# })
# st.dataframe(df)

# st.metric(label="Total Rows", value=str(len(df))+' rub')

# pressed = st.button("press me")
# print(pressed)         

# # Charts 
# chart_data = pd.DataFrame(
#     np.random.rand(20, 3), 
#     columns=['A', 'B', 'C']
# )

# st.subheader("Area Chart")
# st.area_chart(chart_data)
# st.bar_chart(chart_data)
         
# st.balloons() # пустить шарики, типо успешно все завершено