from src.prepare import prepare
from src.featurization import featurize
from src.train import train
from src.evaluate import evaluate

if __name__ == "__main__":
    prepare(run=True)
    featurize(run=True)
    train(run=True)
    evaluate(run=True)

