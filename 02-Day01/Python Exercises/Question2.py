# Write a program which can compute the factorial of a given numbers.

# First try to solve the problem yourself

# Solution

n = int(input())

fact = 1
for i in range(2,n+1):
	fact *= i # same as (fact = fact*i)

print(fact)