#include <stdio.h>
static unsigned int helper(unsigned int v) {
    return v * 7 + 3;
}
int main() {
    unsigned int total = 0;
    for (int i = 0; i < 20; i++) {
        if (i % 2 == 0) total += helper(i);
        else total += i * 2;
    }
    printf("%u\n", total);
    return 0;
}
