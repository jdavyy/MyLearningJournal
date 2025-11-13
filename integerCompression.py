
def compress(x: list) -> str:
    final = ''
    starter = x[0]
    for i in range(len(x)-1):
        if (x[i+1] - 1) == x[i]:
            continue
        else: 
            final += f'{starter}-{x[i]},'
            starter = x[i+1]
    starter = x[-1]
    final += f'{starter}'        
    return final






def main():
    newlist = [1, 2, 3, 6, 7, 9, 10, 11, 15]
    print(compress(newlist))

if __name__ == "__main__":
    main()