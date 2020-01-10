from argparse import ArgumentParser

from people_counter import PeopleCounter


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-p", "--prototxt", type=str, default='train/mobile-net-ssd.prototxt')
    parser.add_argument("-m", "--model", type=str, default='train/mobile-net-ssd.caffemodel')
    parser.add_argument("-s", "--skip-frames", type=int, default=30)
    parser.add_argument("-c", "--confidence", type=float, default=0.5)
    parser.add_argument("-d", "--distance", type=float, default=30.0)
    args = parser.parse_args()
    pc = PeopleCounter(
        args.input,
        args.output,
        args.prototxt,
        args.model,
        args.skip_frames,
        args.confidence,
        args.distance
    )
    pc.init()
    pc.start()


if __name__ == '__main__':
    main()
