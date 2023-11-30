import sys
from io import StringIO

print("Hello World")
# Temporarily redirect standard output to capture the print statement
old_stdout = sys.stdout
sys.stdout = StringIO()

# The print statement you want to capture
print(1 + 1)

# Get the output and revert stdout to its original setting
output = sys.stdout.getvalue()
sys.stdout = old_stdout

# Write the output to a file
with open("output.txt", "a") as file:
    print(output)
    file.write(output)
