#include <stdio.h>
static unsigned int compute(unsigned int a, unsigned int b) {
    return (a ^ b) + (a * b);
}
int main() {
    unsigned int res = compute(15, 27);
    printf("%u\n", res);
    return 0;
}
