#include <stdio.h>
int main() {
    unsigned int a = 1234, b = 5678, c = 9012;
    unsigned int res = (a * b) / (c | 1) + (a ^ b ^ c);
    printf("%u\n", res);
    return 0;
}
