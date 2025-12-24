import os
from datetime import datetime

import httpx
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Модель успешно обучена")
    return {"model": model, "X_test": X_test, "y_test": y_test, "dataset": iris}


def evaluate_model(**kwargs):
    data = kwargs["ti"].xcom_pull(task_ids="train_model")

    model = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    iris = data["dataset"]

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy < 0.95:
        raise AirflowFailException(
            f"Значение метрики accuracy слишком мало: {accuracy}. Не деплоим модель."
        )

    print(f"Точность модели (Accuracy): {accuracy:.2%}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Модель оценена, метрики в норме")


def deploy_model():
    print("Модель выведена в продакшен")


def send_telegram_message():
    token = os.environ["TG_TOKEN"]
    chat_id = os.environ["TG_CHATID"]
    model_version = os.environ["MODEL_VERSION"]
    message = f"Новая модель в продакшене! Версия {model_version}"
    httpx.get(
        f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    )
    print("Сообщение отправлено")


with DAG(
    dag_id="ml_retrain_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    deploy = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    notify = PythonOperator(
        task_id="notify_success", python_callable=send_telegram_message
    )

    train >> evaluate >> deploy >> notify
