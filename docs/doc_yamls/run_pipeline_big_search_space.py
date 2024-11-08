
def example_pipeline(learning_rate, optimizer, epochs, batch_size, dropout_rate):
    model = initialize_model(dropout_rate)
    training_objective_to_minimize = train_model(model, optimizer, learning_rate, epochs, batch_size)
    evaluation_objective_to_minimize = evaluate_model(model)
    return {"objective_to_minimize": evaluation_objective_to_minimize, "training_objective_to_minimize": training_objective_to_minimize}
