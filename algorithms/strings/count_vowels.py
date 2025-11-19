"""
This file is me working on basic python programming skills, towars the bottom are more efficient hash map problems I worked on.
Everything is still very simple.
"""

#counts the number of upper and lowercase vowels in a string
def count_vowels(x: str):
    Uppercount = 0
    Lowercount = 0
    for i in range(len(x)):
        if x[i] in 'AEIOU':
            Uppercount += 1
        elif x[i] in 'aeiou':
            Lowercount += 1

    print(f'# of Uppercase vowels: {Uppercount}')
    print(f'# of Lowercase vowels: {Lowercount}')

#replaces vowels found in a string with other characters.
def replace_vowels(x: str)-> str:
    newString = ''
    for i in range(len(x)):
        if x[i] in 'AEIOUaeiou':
            newString += '*'
        else:
            newString += x[i]
    return newString

#returns the indexes of vowels found in a string
def returnIndexesOfVowel(x: str) -> list:
    newList = []
    for i in range(len(x)):
        if x[i] in 'AEIOUaeiou':
            newList.append(i)
    return newList

#counts the number of words in a string not using any python helper functions like split() or remove()
def countWords(x: str) -> int:
    count = 0
    if (len(x)) ==0:
        return count
    if x[0] != ' ':
        count += 1
    for i in range(1, len(x)):
        if x[i] not in ' ' and x[i-1] in ' ':
            count += 1
    return count

#return the first character in the string that appears exactly once
def first_unique_char(x: str) -> str:
    counts = {}
    for ch in x:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in x:
        if counts[ch] == 1:
            return ch
        
    return ""

#returns the word that is most frequent in the string
def frequentWord(x: str) -> str:
    listOfWords = x.split()
    if not listOfWords:
        return ""
    counts = {}
    for word in listOfWords:
        counts[word] = counts.get(word, 0) + 1
    maxv = max(counts.values())
    for word in listOfWords:
        if counts[word] == maxv:
            return word

#returns true or false based on whether the inputs are anagrams or not
def are_anagrams(a: str, b: str)-> bool:
    aStr = a.lower().replace(" ", "")
    bStr = b.lower().replace(" ", "")

    if not aStr or not bStr:
        raise TypeError("Empty String or other isssue")
    hash_a = {}
    hash_b = {}
    for letter in aStr:
        hash_a[letter] = hash_a.get(letter, 0) + 1
    for letter in bStr:
        hash_b[letter] = hash_b.get(letter, 0) + 1
    return hash_a == hash_b
        

        
#main function holds tests
def main():
    newlist = "hello my name is harald"
    words = "i am testing testing testing this new function here here. "
    print(replace_vowels(newlist))
    print(returnIndexesOfVowel(newlist))
    print(countWords(newlist))
    print(first_unique_char(newlist))
    print(frequentWord(words))
    print(are_anagrams('table', 'labed'))

if __name__ == "__main__":
    main()
