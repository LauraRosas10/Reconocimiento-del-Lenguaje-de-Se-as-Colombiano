import sys

from landmark_recognition.train_model import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--architecture", "lstm"]
    main()
