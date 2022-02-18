from src.evaluate import evaluate
from src.featurization import featurize
from src.prepare import prepare
from src.train import train

if __name__ == "__main__":
    prepare(run=True)
    featurize(run=True)
    train(run=True)
    evaluate(run=True)
