"""Module to rearrange string so no two adjacent letters are the same."""

from string import ascii_uppercase as UP


def rearrange_string(s: str) -> str:
    '''
        input : string containing lowercase alphabets
        output: rearranged string with no two adjacent letters being the same
                -1 if this is not possible.
    '''
    nos = [0] * 26

    for x in s:
        nos[ord(x) - ord('A')] += 1

    prev = 'a'
    ans = []

    for i in range(len(s)):
        possible = ""
        found = False
        for j in range(26):
            if not found and UP[j] != prev and nos[j] > 0:
                possible = UP[j]
                found = True
            if nos[j] > (len(s) - i) // 2:  # will become impossible
                possible = UP[j]

        if possible == "":
            return "-1"

        ans.append(possible)
        nos[ord(possible) - ord('A')] -= 1
        prev = possible

    for i in range(len(s) - 1):
        if ans[i] == ans[i + 1]:
            return "-1"

    return "".join(ans)


if __name__ == "__main__":
    input_string = input()
    print(rearrange_string(input_string))
