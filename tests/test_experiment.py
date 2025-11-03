import os
import signal
import subprocess
import sys
import time


def a_stubborn_child():
    """A child process that spawns another process."""
    print("Spawning a grandchild process...")
    # This grandchild will just sleep for a long time
    subprocess.Popen(["sleep", "60"])
    print("Grandchild spawned.")


def handle_sigterm(sig, frame):
    print("CHILD: Received SIGTERM, but I'm ignoring it for 10 seconds!")
    time.sleep(10)
    print("CHILD: Okay, now I'm exiting.")
    sys.exit(0)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if scenario == "stubborn":
        # This process will ignore SIGTERM for a while
        signal.signal(signal.SIGTERM, handle_sigterm)
        print("CHILD (stubborn): Running. PID:", os.getpid())
    elif scenario == "spawner":
        # This process will start a grandchild
        a_stubborn_child()
        print("CHILD (spawner): Running. PID:", os.getpid())
    else:
        # Normal process
        print("CHILD (normal): Running. PID:", os.getpid())

    # Main loop
    count = 0
    while True:
        print(f"CHILD: ...alive... {count}")
        time.sleep(2)
        count += 1
