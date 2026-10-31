"""Record keyboard/mouse input; F8 start/pause/resume, F9 stop and save.
Run: py macro_record.py
Output: recordings/input_<timestamp>.json and .csv
"""


def main():
    from tailstopia_input_recorder import main as record_inputs
    record_inputs()


if __name__ == "__main__":
    main()
