

def example_pipeline(learning_rate, optimizer, epochs):
    model = initialize_model()
    training_loss = train_model(model, optimizer, learning_rate, epochs)
    evaluation_loss = evaluate_model(model)
    return {"loss": evaluation_loss, "training_loss": training_loss}
