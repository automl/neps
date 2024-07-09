
def example_pipeline(learning_rate, optimizer, epochs, batch_size, dropout_rate):
    model = initialize_model(dropout_rate)
    training_loss = train_model(model, optimizer, learning_rate, epochs, batch_size)
    evaluation_loss = evaluate_model(model)
    return {"loss": evaluation_loss, "training_loss": training_loss}
