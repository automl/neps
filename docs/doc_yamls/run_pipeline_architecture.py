from torch import nn


def example_pipeline(architecture, optimizer, learning_rate):
    in_channels = 3
    base_channels = 16
    n_classes = 10
    out_channels_factor = 4

    # E.g., in shape = (N, 3, 32, 32) => out shape = (N, 10)
    model = architecture.to_pytorch()
    model = nn.Sequential(
        nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(base_channels),
        model,
        nn.BatchNorm2d(base_channels * out_channels_factor),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(base_channels * out_channels_factor, n_classes),
    )
    training_objective_to_minimize = train_model(model, optimizer, learning_rate)
    evaluation_objective_to_minimize = evaluate_model(model)
    return {"objective_to_minimize": evaluation_objective_to_minimize, "training_objective_to_minimize": training_objective_to_minimize}
