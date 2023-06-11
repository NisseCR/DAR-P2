from preprocessing import preprocess
from modelling import model


def main():
    preprocess('product_title')
    model()


if __name__ == '__main__':
    main()
