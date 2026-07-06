from datetime import datetime
import pytz

def print_log(msg: str):
    """
    Utility to print log with message.

    Args:
        msg: A text message to be print in log.
    """
    print(datetime.now().strftime('%d-%m-%Y %H:%M:%S'),f'- {msg}')

if __name__ == '__main__':
    print_log('test')