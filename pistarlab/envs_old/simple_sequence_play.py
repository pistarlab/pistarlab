from pistarlab.envs.simple_sequence import SimpleSequence


def main():
    game = SimpleSequence(enable_render=True)
    game.reset(human=True)

    while (True):
        game.render()
        good_value = False
        while not good_value:
            value = input().strip()
            if value == 'x':
                exit(0)
            if value not in ("0", "1", "2"):
                print("Not valid input, please enter '0' for no action, '1' for odd, '2' for even.  Enter 'x' to exit")
            else:
                good_value = True
                value = int(value)
        game.step(action=value, human=True)


if __name__ == '__main__':
    main()
