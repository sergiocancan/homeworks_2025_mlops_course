import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# import mlflow
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-homework2_2_sc")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def ensure_dense(X):
    """Convierte a denso si el objeto tiene el método `.toarray()` (sparse matrices)."""
    return X.toarray() if hasattr(X, 'toarray') else X

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Validación de tipo para asegurar que MLflow no falle
        X_train = ensure_dense(X_train)
        X_val = ensure_dense(X_val)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}")

if __name__ == '__main__':
    mlflow.sklearn.autolog()    
    run_train()
