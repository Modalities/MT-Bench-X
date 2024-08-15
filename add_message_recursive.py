from pathlib import Path

# Directory to search
directory = Path("/raid/s3/opengptx/alexw/mtbenchx/src/mtbenchx/fastchat")

# Message to be added
message = """# This file is part of a project that has been adapted from FastChat.
# For more information, visit: https://github.com/lm-sys/FastChat
# Distributed under the Apache License, Version 2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for more details.

"""


def add_message_to_file(file_path):
    content = file_path.read_text()
    file_path.write_text(message + content)


# Walk through the directory recursively
for file_path in directory.rglob("*"):
    if file_path.is_file():
        add_message_to_file(file_path)

print(f"Message added to the top of each file in {directory}")
