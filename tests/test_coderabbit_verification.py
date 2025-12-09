# Test module for CodeRabbit verification
# This file will be removed after testing

def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    total = 0
    for n in numbers:
        total = total + n
    return total

def find_max(numbers):
    """Find the maximum value in a list."""
    if len(numbers) == 0:
        return None
    max_val = numbers[0]
    for n in numbers:
        if n > max_val:
            max_val = n
    return max_val

# TODO: Add error handling for edge cases
def divide(a, b):
    """Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of division
        
    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def multiply(a, b):
    """Multiply two numbers."""
    return a * b
