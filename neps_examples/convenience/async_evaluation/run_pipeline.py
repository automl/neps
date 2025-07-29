# run_pipeline.py ----------------------------------------------------------
import argparse, time, json, torch, torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import neps

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pipeline_id", type=int, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        parser.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
        parser.add_argument("--root_directory", type=str, required=True)
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

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
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
        if args.pipeline_id == 1:
            raise ValueError("This is a dummy error to test the exception handling in the evaluation script.") 
    except Exception as e:
        print(f"Error during evaluation: {e}")
        user_result = dict(objective_to_minimize=None, cost=None, exception= e)
    neps.save_pipeline_results(
        user_result=user_result,
        pipeline_id=args.pipeline_id,
        root_directory=Path(args.root_directory),
    )
