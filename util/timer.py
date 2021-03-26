import time

start_time = None


def start():
    global start_time
    start_time = time.time()


def end():
    print(f"Runtime: {time.time() - start_time} [s]")
