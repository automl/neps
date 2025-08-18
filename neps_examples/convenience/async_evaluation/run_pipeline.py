# run_pipeline.py ----------------------------------------------------------
import argparse, time, json, torch, torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import neps
from neps.plot.tensorboard_eval import tblogger


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pipeline-id", type=int, required=True)
        parser.add_argument("--learning-rate", type=float, required=True)
        parser.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
        parser.add_argument("--root-directory", type=str, required=True)
        parser.add_argument("--pipeline-directory", type=str, required=True)
        parser.add_argument("--previous-pipeline-directory", type=str, required=True)
        args = parser.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        start_eval = time.time()
        # toy 1‑layer model
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = (torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam)(
            model.parameters(), lr=args.learning_rate
        )

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/tmp",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("/tmp", train=False, transform=transforms.ToTensor()),
            batch_size=1024,
        )

        writer = tblogger.ConfigWriter(write_summary_incumbent=True, 
                                       root_directory=args.root_directory,
                                       pipeline_directory=args.pipeline_directory,
                                       previous_pipeline_directory=args.previous_pipeline_directory)

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            writer.add_scalar("train/loss", loss.item(), i)
            if i % 10 == 0:
                print(f"Pipeline ID {args.pipeline_id} - Iteration {i}: Loss = {loss.item()}")
            loss.backward()
            opt.step()

        time.sleep(2)  # simulate some time for training

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        user_result = dict(objective_to_minimize=1-acc, cost=time.time() - start_eval)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        user_result = dict(objective_to_minimize=None, cost=None, exception= e)
    neps.save_pipeline_results(
        user_result=user_result,
        pipeline_id=args.pipeline_id,
        root_directory=Path(args.root_directory),
    )
