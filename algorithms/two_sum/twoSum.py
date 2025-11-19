"""
Problem:
Given a list of integers nums and an integer target, return the indices of the two numbers that add up to target.
You must return the first valid pair you find.
Assume exactly one solution exists.

Example
nums = [2, 7, 11, 15]
target = 9

Output: [0, 1]
"""
#result using O(n^2) time complexion
def twoSum(nums: list, target: int) -> list:
    final = []
    for i in range(0, len(nums)):
        for x in range(0, len(nums)):
            if (nums[i] + nums[x]) == target:
                final.append(i)
                final.append(x)
                return final
        
#result using hashmap O(n) time complexion
def twoSumHash(nums: list, target: int) -> list[int]:
    newlist = {}
    for i in range(len(nums)):
        num = nums[i]
        diff = target - num
        if diff in newlist:
            return [newlist[diff], i]
        newlist[num] = i
    return []

#easy contains duplicate function
def containsDup(numbers: list) -> bool:
    #lets make a hashmap
    seen = {}
    for number in numbers:
        seen[number] = seen.get(number, 0) + 1
        if seen[number] >= 2:
            return True
        
    return False


def main():

    nums = [2,7,11,15]
    target = 9
    print(twoSum(nums,target))

    numss = [2,7,11,15]
    target = 18
    print(twoSumHash(numss,target))

    nums1 = [1,2,3,1]
    print(containsDup(nums1))

if __name__ == "__main__":
    main()
