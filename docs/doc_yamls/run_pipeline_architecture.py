from torch import nn


def example_pipeline(architecture, optimizer, learning_rate):
    in_channels = 3
    base_channels = 16
    n_classes = 10
    out_channels_factor = 4

    # E.g., in shape = (N, 3, 32, 32) => out shape = (N, 10)
    model = architecture.to_pytorch()
    training_loss = train_model(model, optimizer, learning_rate)
    evaluation_loss = evaluate_model(model)
    return {"loss": evaluation_loss, "training_loss": training_loss}
