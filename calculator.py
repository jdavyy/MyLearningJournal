import sys

def add(x: int, y: int) -> int:
    return x + y

def subtract(x: int, y: int) -> int:
    return x - y

def mult(x: int, y: int) -> int:
    return x * y

def div(x: int, y: int) -> float:
    return x / y

try:
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
except ValueError:
    print("Please enter valid numbers only.")
    exit()

OpType = input("Enter the operation type: add, subtract, mult, div: ")
if OpType == 'add':
    print(add(num1, num2))
elif OpType == 'subtract':
    print(subtract(num1, num2))
elif OpType == 'mult':
    print(mult(num1, num2))
elif OpType == 'div':
    if num2 == 0:
        print("Cannot divide by zero.")
        exit()
    print(div(num1, num2))
else: 
    print("Operation type could not be determined please try again.")
    #ideally it would loop over and then ask for new input for the operation (input validation loop)
    #Can be added for a quick fix for later



