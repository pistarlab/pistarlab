from pistarlab.envs.cliff_walking import CliffWalkingEnv


def main():
    game = CliffWalkingEnv()
    game.reset()

    while (True):
        game.render(mode='human')
        good_value = False
        while not good_value:
            value = input().strip()
            if value == 'x':
                exit(0)
            if value not in ("0", "1", "2", "3"):
                print("Not valid input,")
            else:
                good_value = True
                value = int(value)
        step_results = game.step(a=value)
        print(step_results)


if __name__ == '__main__':
    main()
