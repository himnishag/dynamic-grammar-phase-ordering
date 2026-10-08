#include <stdio.h>
int main() {
    unsigned int x = 42, y = 100, z = 0;
    if (x > 20) z = x * 2;
    else z = y + 5;
    if (z % 2 == 0) z += 10;
    printf("%u\n", z);
    return 0;
}
