from models import build_baseline_cnn, build_densenet121


def main():
    print("Testing Baseline CNN...")
    baseline = build_baseline_cnn()
    baseline.summary()

    print("\nTesting DenseNet121...")
    densenet = build_densenet121()
    densenet.summary()

    print("\nBoth models were created successfully!")


if __name__ == "__main__":
    main()