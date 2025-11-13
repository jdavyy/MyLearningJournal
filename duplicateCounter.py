"""
Simple function to remove duplicates from a list
"""

def remove(x: list) -> list:
    final = []
    for item in x:
        if item not in final:
            final.append(item)
    return final


def main():
    newlist = [1, 2, 4, 4, 3, 2, 1, 3, 2]
    print(remove(newlist))
    

if __name__ == "__main__":
    main()
