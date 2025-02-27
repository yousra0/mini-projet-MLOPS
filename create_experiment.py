import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "Churn Prediction Experiment"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"✅ Expérience créée avec l'ID : {experiment_id}")
else:
    print(f"🔄 L'expérience '{experiment_name}' existe déjà avec l'ID : {experiment.experiment_id}")
