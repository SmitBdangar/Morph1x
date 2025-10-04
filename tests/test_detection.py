def char():
    cha = input("Enter a character: ")
    li = ["g", "r", "r", "e"]
    num = 0
    for i in li:
        if i == cha:
            print("found")
            num += 1
        else:
            print(i)

    print(f"total {cha} in currunt cotext is {num}")


char()
